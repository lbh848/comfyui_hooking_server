"""
bubble_match - 말풍선 모드: 발화자(NAME) ↔ 감지된 얼굴 매칭

speak 텍스트의 NAME 들과 삽화에서 감지된 얼굴들을 ViT-L/14 임베딩 코사인 유사도로 매칭한다.

모든 경우에 각 얼굴 임베딩 ↔ 각 NAME 캐릭터 임베딩 코사인 유사도를
계산하고 전체 조합의 최적 1:1 배정을 구한다. NAME/얼굴이 각각 하나뿐이어도 임계치를
우회하지 않으며, 임계치 미달 시 미배정하고 사유를 로그에 남긴다.

매칭 풀은 speak 에 등장한 NAME 들로 한정한다(NAME: 발화 형식이므로 주어진 NAME 만 사용).
"""

import traceback

import numpy as np

# 매칭 신뢰 임계치 (L2 정규화 코사인 유사도). 같은 캐릭터면 보통 0.7 이상.
_MATCH_THRES_DEFAULT = 0.55


def _cosine(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def _optimal_assignment(similarities, match_thres):
    """임계치를 통과한 NAME↔얼굴의 최대 cardinality·최대 유사도 배정.

    Hungarian 알고리즘으로 모든 NAME에 실제 얼굴 또는 전용 dummy 열을 배정한다.
    유효한 실제 얼굴에는 충분한 cardinality 보너스를 더해 먼저 배정 인원 수를
    최대화하고, 같은 인원 수 안에서는 코사인 유사도 총합을 최대화한다.

    Returns:
        {name_row_index: (face_index, similarity)}
    """
    row_count = len(similarities)
    face_count = max((len(row) for row in similarities), default=0)
    if row_count == 0 or face_count == 0:
        return {}

    # 실제 얼굴 열 + 각 NAME이 안전하게 미배정될 수 있는 dummy 열.
    column_count = face_count + row_count
    cardinality_bonus = float(max(row_count, face_count) + 1)
    costs = []
    for row in similarities:
        padded = list(row) + [None] * (face_count - len(row))
        benefits = [
            cardinality_bonus + float(sim)
            if sim is not None and sim >= match_thres
            else -cardinality_bonus
            for sim in padded
        ]
        benefits.extend([0.0] * row_count)
        costs.append([-benefit for benefit in benefits])

    # Rectangular Hungarian (rows <= columns), 1-based 잠재치 구현.
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    p = [0] * (column_count + 1)
    way = [0] * (column_count + 1)
    for i in range(1, row_count + 1):
        p[0] = i
        minv = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, column_count + 1):
                if used[j]:
                    continue
                cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(column_count + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_to_column = {}
    for column in range(1, column_count + 1):
        if p[column]:
            row_to_column[p[column] - 1] = column - 1

    assigned = {}
    for row_index, face_index in row_to_column.items():
        if face_index >= face_count:
            continue
        sim = similarities[row_index][face_index]
        if sim is not None and sim >= match_thres:
            assigned[row_index] = (face_index, float(sim))
    return assigned


def match_speakers_to_faces(segments, faces, bot_name, match_thres=_MATCH_THRES_DEFAULT):
    """발화자(NAME) ↔ 얼굴 매칭.

    Args:
        segments: parse_speak() 결과 [{speaker, text, type, emotion}, ...]
        faces: detect_faces() 결과 [{box, conf}, ...]
        bot_name: 봇 이름 (캐릭터 임베딩 조회용)
        match_thres: 코사인 유사도 임계치

    Returns:
        [{segment, face_box|None, char_name|None, sim|None}, ...] — segments 순서 보존.
    """
    try:
        from modes.face_embedder import embed_face_crop, get_char_embedding
    except Exception as e:
        print(f"[BUBBLE_MATCH] 의존 모듈 로드 실패: {e}")
        traceback.print_exc()
        return [{"segment": s, "face_box": None, "char_name": None, "sim": None} for s in segments]

    # 발화자(NAME) 목록 — None(독백) 제외, 순서 보존 중복 제거
    names = []
    for s in segments:
        sp = (s or {}).get("speaker")
        if sp and sp not in names:
            names.append(sp)

    if not names:
        print("[BUBBLE_MATCH] SPEAK에 NAME 형식의 발화자가 없어 전부 미배정")
        return [{"segment": s, "face_box": None, "char_name": None, "sim": None} for s in segments]

    # 얼굴이 하나도 없으면 전부 미배정
    if not faces:
        print("[BUBBLE_MATCH] 감지된 얼굴 0건 — 전부 미배정")
        return [{"segment": s, "face_box": None, "char_name": None, "sim": None} for s in segments]

    # ── 각 NAME 의 캐릭터 임베딩 ──
    name_embs = {}
    for n in names:
        emb = get_char_embedding(bot_name, n)
        if emb is None:
            print(f"[BUBBLE_MATCH] 캐릭터 임베딩 없음: {bot_name}/{n} → 이 NAME 은 미배정")
        else:
            name_embs[n] = emb

    if not name_embs:
        print("[BUBBLE_MATCH] 사용 가능 캐릭터 임베딩 0건 — 전부 미배정")
        return [{"segment": s, "face_box": None, "char_name": None, "sim": None} for s in segments]

    # ── 각 얼굴 임베딩 ──
    # 주의: faces 의 box 는 원본 이미지 좌표. 호출자가 같은 PIL.Image 를 전달해야 한다.
    # 여기서는 image 없이는 임베딩 불가 → faces 에 'image' 가 있으면 사용, 아니면 미배정.
    image = None
    for f in faces:
        if f.get("image") is not None:
            image = f["image"]
            break
    if image is None:
        print("[BUBBLE_MATCH] faces 에 image 없음 — 얼굴 임베딩 불가, 전부 미배정 "
              "(호출자가 faces[i]['image'] 에 PIL.Image 를 넣어야 함)")
        return [{"segment": s, "face_box": None, "char_name": None, "sim": None} for s in segments]

    face_embs = []
    for i, f in enumerate(faces):
        emb = embed_face_crop(image, f["box"])
        face_embs.append(emb)
        if emb is None:
            print(f"[BUBBLE_MATCH] 얼굴 {i} 임베딩 실패: {f['box']}")

    # ── 유사도 매트릭스 → 전역 최적 1:1 배정 ──
    # 후보를 개별 최고점 순서로 고르면 한 캐릭터가 다른 캐릭터의 유일한 얼굴을
    # 먼저 차지할 수 있다. 전체 조합에서 배정 인원 수와 유사도 합을 차례로 최대화한다.
    embedded_names = list(name_embs)
    similarities = []
    for n in embedded_names:
        row = []
        for fi, fe in enumerate(face_embs):
            sim = _cosine(name_embs[n], fe) if fe is not None else None
            row.append(sim)
            face_conf = faces[fi].get("conf")
            conf_text = f"{float(face_conf):.3f}" if face_conf is not None else "없음"
            sim_text = f"{sim:.3f}" if sim is not None else "계산불가"
            print(
                f"[BUBBLE_MATCH] 유사도: {n} ↔ 얼굴{fi} "
                f"(sim={sim_text}, yolo_conf={conf_text}, box={faces[fi]['box']})"
            )
        similarities.append(row)

    optimal = _optimal_assignment(similarities, match_thres)
    name_to_face = {}
    for name_index, (fi, sim) in optimal.items():
        n = embedded_names[name_index]
        name_to_face[n] = (fi, sim)
        print(f"[BUBBLE_MATCH] 배정: {n} ↔ 얼굴{fi} (box={faces[fi]['box']}, sim={sim:.3f})")

    for n in names:
        if n not in name_to_face and n in name_embs:
            print(f"[BUBBLE_MATCH] 미배정(NAME 임계치 미달/면제): {n}")

    # 세그먼트별 결과 조립
    results = []
    for s in segments:
        sp = (s or {}).get("speaker")
        if sp and sp in name_to_face:
            fi, sim = name_to_face[sp]
            results.append({"segment": s, "face_box": faces[fi]["box"],
                            "char_name": sp, "sim": sim})
        else:
            results.append({"segment": s, "face_box": None, "char_name": sp, "sim": None})
    return results
