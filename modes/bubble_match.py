"""
bubble_match - 말풍선 모드: 발화자(NAME) ↔ 감지된 얼굴 매칭

speak 텍스트의 NAME 들과 삽화에서 감지된 얼굴들을 ViT-L/14 임베딩 코사인 유사도로 매칭한다.

모든 경우에 각 얼굴 임베딩 ↔ 각 NAME 캐릭터 임베딩 코사인 유사도를
계산하고 그리디 배정한다(중복 방지). NAME/얼굴이 각각 하나뿐이어도 임계치를
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

    # ── 유사도 매트릭스 → 그리디 배정 (NAME 마다 서로 다른 얼굴) ──
    # (name, face_idx, sim) 후보 전부 생성 후 sim 내림차순 그리디.
    cands = []
    for n, ne in name_embs.items():
        for fi, fe in enumerate(face_embs):
            if fe is None:
                continue
            cands.append((_cosine(ne, fe), n, fi))
    cands.sort(key=lambda x: x[0], reverse=True)

    name_to_face = {}   # name -> (face_idx, similarity)
    used_faces = set()
    for sim, n, fi in cands:
        if n in name_to_face or fi in used_faces:
            continue
        if sim < match_thres:
            continue
        name_to_face[n] = (fi, sim)
        used_faces.add(fi)
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
