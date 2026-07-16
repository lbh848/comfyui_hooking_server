"""
bubble_match - 말풍선 모드: 발화자(NAME) ↔ 감지된 얼굴 매칭

speak 텍스트의 NAME 들과 삽화에서 감지된 얼굴들을 ViT-L/14 임베딩 코사인 유사도로 매칭한다.

모든 경우에 각 얼굴 임베딩 ↔ 각 NAME 캐릭터 임베딩 코사인 유사도를
계산하고 전체 조합의 최적 1:1 배정을 구한다. NAME/얼굴이 각각 하나뿐이어도 임계치를
우회하지 않으며, 임계치 미달 시 미배정하고 사유를 로그에 남긴다.

매칭 풀은 speak 에 등장한 NAME 들로 한정한다(NAME: 발화 형식이므로 주어진 NAME 만 사용).
"""

import json
import os
import traceback

import numpy as np

# 매칭 신뢰 임계치 (L2 정규화 코사인 유사도). 같은 캐릭터면 보통 0.7 이상.
_MATCH_THRES_DEFAULT = 0.55
_BOT_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "asset_data",
    "bot.json",
)
_MISSING_PROJECT_CHARACTER = "missing_project_character"


def _project_character_name_map(bot_name):
    """봇 프로젝트에 등록된 캐릭터 이름을 casefold 키로 읽는다.

    ``None``은 메타데이터를 읽지 못했거나 대상 봇을 확인하지 못했다는 뜻이다.
    이때는 캐릭터가 없다고 단정하지 않고 기존 임베딩 조회 경로를 유지한다.
    """
    target_bot = str(bot_name or "").strip()
    if not target_bot:
        print("[BUBBLE_MATCH] 봇 이름 없음 - 프로젝트 캐릭터 등록 여부 확인 불가")
        return None
    if not os.path.isfile(_BOT_DATA_FILE):
        print(
            f"[BUBBLE_MATCH] 봇 메타데이터 파일 없음 - "
            f"프로젝트 캐릭터 등록 여부 확인 불가: {_BOT_DATA_FILE}"
        )
        return None
    try:
        with open(_BOT_DATA_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as e:
        print(f"[BUBBLE_MATCH] 봇 메타데이터 로드 실패({_BOT_DATA_FILE}): {e}")
        traceback.print_exc()
        return None

    bots = data.get("bots") if isinstance(data, dict) else None
    if not isinstance(bots, list):
        print(
            f"[BUBBLE_MATCH] 봇 메타데이터 형식 오류 - bots 목록 없음: "
            f"{_BOT_DATA_FILE}"
        )
        return None

    bot = next(
        (
            item for item in bots
            if isinstance(item, dict)
            and str(item.get("name") or "").strip().casefold() == target_bot.casefold()
        ),
        None,
    )
    if bot is None:
        print(
            f"[BUBBLE_MATCH] 봇 프로젝트를 메타데이터에서 찾지 못함: "
            f"{target_bot} - 캐릭터 없음으로 단정하지 않음"
        )
        return None

    characters = bot.get("characters")
    if not isinstance(characters, list):
        print(f"[BUBBLE_MATCH] 봇 캐릭터 목록 형식 오류: {target_bot}")
        return None
    result = {}
    for character in characters:
        if not isinstance(character, dict):
            print(
                f"[BUBBLE_MATCH] 잘못된 캐릭터 항목 건너뜀: "
                f"bot={target_bot}, value={character!r}"
            )
            continue
        name = str(character.get("name") or "").strip()
        if not name:
            print(f"[BUBBLE_MATCH] 이름 없는 캐릭터 항목 건너뜀: bot={target_bot}")
            continue
        result[name.casefold()] = name
    return result


def _unmatched_segment_results(segments, missing_project_names=None):
    """미배정 결과를 만들되 프로젝트에 없는 NAME의 사유만 보존한다."""
    missing_keys = {
        str(name).casefold() for name in (missing_project_names or ())
        if str(name).strip()
    }
    results = []
    for segment in segments:
        speaker = (segment or {}).get("speaker")
        item = {
            "segment": segment,
            "face_box": None,
            "char_name": speaker,
            "sim": None,
        }
        if speaker and str(speaker).casefold() in missing_keys:
            item["unmatched_reason"] = _MISSING_PROJECT_CHARACTER
        results.append(item)
    return results


def _cosine(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def _optimal_assignment(similarities, match_thres, ranking_scores=None,
                        forbidden_pairs=None):
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
    ranking_scores = ranking_scores if ranking_scores is not None else similarities
    forbidden_pairs = set(forbidden_pairs or ())
    costs = []
    for row_index, row in enumerate(similarities):
        padded = list(row) + [None] * (face_count - len(row))
        rank_row = list(ranking_scores[row_index]) + [None] * (
            face_count - len(ranking_scores[row_index])
        )
        benefits = [
            cardinality_bonus + float(rank_row[face_index])
            if (
                sim is not None
                and rank_row[face_index] is not None
                and sim >= match_thres
                and (row_index, face_index) not in forbidden_pairs
            )
            else -cardinality_bonus
            for face_index, sim in enumerate(padded)
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


def _assignment_score(assignment, scores):
    """배정의 ranking score 합계."""
    return sum(float(scores[row][face]) for row, (face, _sim) in assignment.items())


def _assignment_ambiguity_gap(similarities, ranking_scores, match_thres, best):
    """최적 배정과 같은 인원 수의 차선 배정 사이 평균 점수 차이를 반환한다."""
    if not best:
        return None
    best_score = _assignment_score(best, ranking_scores)
    alternative_scores = []
    for row, (face, _sim) in best.items():
        alternative = _optimal_assignment(
            similarities,
            match_thres,
            ranking_scores=ranking_scores,
            forbidden_pairs={(row, face)},
        )
        if len(alternative) == len(best):
            alternative_scores.append(_assignment_score(alternative, ranking_scores))
    if not alternative_scores:
        return None
    return max(0.0, (best_score - max(alternative_scores)) / max(1, len(best)))


def _face_boxes_overlap(first, second):
    """두 RAW 얼굴 박스가 면적을 가진 교집합을 공유하는지 반환한다."""
    try:
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
    except (TypeError, ValueError) as e:
        print(
            f"[BUBBLE_MATCH] 얼굴 박스 교차 판정 실패: "
            f"first={first!r}, second={second!r}, error={e}"
        )
        traceback.print_exc()
        return False
    return (
        min(ax2, bx2) > max(ax1, bx1)
        and min(ay2, by2) > max(ay1, by1)
    )


def _sequential_overlap_assignment(similarities, ranking_scores, match_thres,
                                   face_boxes, ambiguity_margin=0.0):
    """한 캐릭터를 확정할 때마다 겹치는 얼굴 후보를 폐기하고 재배정한다.

    매 단계에서 남은 행/열로 전역 최적 배정을 먼저 구한다. 그 배정 중 선택된
    얼굴에서 다른 캐릭터보다 점수 우위가 가장 큰 쌍을 하나 확정한 뒤, 확정된
    확장 매칭 박스와 면적이 조금이라도 겹치는 모든 후보를 제거한다. 따라서 동일한
    물리 얼굴의 어긋난 박스·포함 박스가 다른 캐릭터에게 재사용되지 않는다.

    Returns:
        (assignment, steps)
        assignment: {원본 캐릭터 행: (원본 얼굴 열, clip_similarity)}
        steps: 확정 순서와 제거된 얼굴 열을 담은 로그용 목록
    """
    row_count = len(similarities or [])
    face_count = len(face_boxes or [])
    active_rows = list(range(row_count))
    active_faces = list(range(face_count))
    assignment = {}
    steps = []
    ambiguity_margin = max(0.0, float(ambiguity_margin or 0.0))

    while active_rows and active_faces:
        sub_similarities = [
            [similarities[row][face] for face in active_faces]
            for row in active_rows
        ]
        sub_rankings = [
            [ranking_scores[row][face] for face in active_faces]
            for row in active_rows
        ]
        sub_assignment = _optimal_assignment(
            sub_similarities,
            match_thres,
            ranking_scores=sub_rankings,
        )
        if not sub_assignment:
            print(
                f"[BUBBLE_MATCH] 순차 배정 중 유효 후보 없음: "
                f"remaining_names={active_rows}, remaining_faces={active_faces}"
            )
            break

        candidates = []
        for sub_row, (sub_face, sim) in sub_assignment.items():
            row = active_rows[sub_row]
            face = active_faces[sub_face]
            rank = ranking_scores[row][face]
            if rank is None:
                print(
                    f"[BUBBLE_MATCH] 순차 배정 rank 없음 - 후보 제외: "
                    f"name_row={row}, face={face}"
                )
                continue
            competitor_scores = [
                ranking_scores[other_row][face]
                for other_row in active_rows
                if other_row != row
                and similarities[other_row][face] is not None
                and similarities[other_row][face] >= match_thres
                and ranking_scores[other_row][face] is not None
            ]
            identity_gap = (
                float(rank) - max(float(score) for score in competitor_scores)
                if competitor_scores else float("inf")
            )
            candidates.append((identity_gap, float(rank), row, face, sim))

        if not candidates:
            print("[BUBBLE_MATCH] 순차 배정에서 확정 가능한 후보 0건")
            break

        # 캐릭터 식별 우위가 큰 쌍을 먼저 확정한다. 같은 우위면 매칭 점수가
        # 높은 쌍을 고른다. 남은 캐릭터가 하나면 identity_gap=inf이다.
        identity_gap, rank, row, face, sim = max(
            candidates,
            key=lambda item: (item[0], item[1], -item[2], -item[3]),
        )
        if (
            len(active_rows) > 1
            and identity_gap < ambiguity_margin
        ):
            print(
                f"[BUBBLE_MATCH] 순차 배정 식별 우위 부족 - 남은 후보 미배정: "
                f"gap={identity_gap:.4f}, 기준={ambiguity_margin:.4f}, "
                f"remaining_names={active_rows}, remaining_faces={active_faces}"
            )
            break

        assignment[row] = (face, sim)
        selected_box = face_boxes[face]
        removed_faces = [
            candidate_face for candidate_face in active_faces
            if _face_boxes_overlap(selected_box, face_boxes[candidate_face])
        ]
        if face not in removed_faces:
            # 자기 박스는 항상 소비한다. 잘못된 좌표라도 같은 얼굴을 재사용하지 않는다.
            removed_faces.append(face)
            removed_faces.sort()
        steps.append({
            "row": row,
            "face": face,
            "sim": sim,
            "rank": rank,
            "identity_gap": identity_gap,
            "removed_faces": removed_faces,
        })
        active_rows = [candidate for candidate in active_rows if candidate != row]
        removed_set = set(removed_faces)
        active_faces = [
            candidate for candidate in active_faces
            if candidate not in removed_set
        ]

    return assignment, steps


def match_speakers_to_faces(segments, faces, bot_name,
                            match_thres=_MATCH_THRES_DEFAULT,
                            appearance_weight=0.4,
                            ambiguity_margin=0.01,
                            detection_confidence_weight=0.05,
                            face_crop_top=2.5,
                            face_crop_bottom=1.0,
                            onnx_device="auto",
                            cpu_threads=0):
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
        from modes.face_embedder import (
            appearance_descriptor,
            appearance_similarity,
            embed_face_crop,
            expanded_face_box,
            extract_face_crop,
            get_char_appearance,
            get_char_embedding,
        )
    except Exception as e:
        print(f"[BUBBLE_MATCH] 의존 모듈 로드 실패: {e}")
        traceback.print_exc()
        return _unmatched_segment_results(segments)

    # 발화자(NAME) 목록 — None(독백) 제외, 순서 보존 중복 제거
    names = []
    for s in segments:
        sp = (s or {}).get("speaker")
        if sp and sp not in names:
            names.append(sp)

    if not names:
        print("[BUBBLE_MATCH] SPEAK에 NAME 형식의 발화자가 없어 전부 미배정")
        return _unmatched_segment_results(segments)

    # ── 각 NAME 의 캐릭터 임베딩 ──
    name_embs = {}
    name_appearances = {}
    missing_project_names = set()
    project_names = _project_character_name_map(bot_name)
    for n in names:
        normalized_name = str(n).strip().casefold()
        if project_names is not None and normalized_name not in project_names:
            missing_project_names.add(n)
            print(
                f"[BUBBLE_MATCH] 봇 프로젝트에 캐릭터 없음: "
                f"{bot_name}/{n} → 무꼬리 빈 공간 배치 대상"
            )
            continue
        canonical_name = (
            project_names.get(normalized_name, n)
            if project_names is not None else n
        )
        if onnx_device in (None, "auto") and int(cpu_threads or 0) == 0:
            emb = get_char_embedding(bot_name, canonical_name)
        else:
            emb = get_char_embedding(
                bot_name,
                canonical_name,
                device=onnx_device,
                cpu_threads=cpu_threads,
            )
        if emb is None:
            print(
                f"[BUBBLE_MATCH] 등록 캐릭터 임베딩 없음/실패: "
                f"{bot_name}/{canonical_name} → 기존 미배정 유지"
            )
        else:
            name_embs[n] = emb
            name_appearances[n] = get_char_appearance(bot_name, canonical_name)

    # 얼굴이 없더라도 프로젝트에 없는 발화자는 무꼬리 폴백으로 전달해야 한다.
    if not faces:
        print("[BUBBLE_MATCH] 감지된 얼굴 0건 - 등록 캐릭터는 미배정")
        return _unmatched_segment_results(segments, missing_project_names)

    if not name_embs:
        print(
            "[BUBBLE_MATCH] 사용 가능 캐릭터 임베딩 0건 - "
            "프로젝트 외 캐릭터만 무꼬리 폴백, 나머지는 미배정"
        )
        return _unmatched_segment_results(segments, missing_project_names)

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
        return _unmatched_segment_results(segments, missing_project_names)

    face_embs = []
    face_appearances = []
    expanded_face_boxes = []
    for i, f in enumerate(faces):
        expanded_box = expanded_face_box(
            image,
            f["box"],
            top_mult=face_crop_top,
            bottom_mult=face_crop_bottom,
        )
        expanded_face_boxes.append(expanded_box)
        embed_kwargs = {
            "top_mult": face_crop_top,
            "bottom_mult": face_crop_bottom,
        }
        if onnx_device not in (None, "auto") or int(cpu_threads or 0) != 0:
            embed_kwargs.update(device=onnx_device, cpu_threads=cpu_threads)
        emb = embed_face_crop(image, f["box"], **embed_kwargs)
        face_embs.append(emb)
        crop = extract_face_crop(
            image,
            f["box"],
            top_mult=face_crop_top,
            bottom_mult=face_crop_bottom,
        )
        face_appearances.append(appearance_descriptor(crop) if crop is not None else None)
        if crop is not None:
            print(
                f"[BUBBLE_MATCH] 얼굴{i} 매칭 크롭: raw={f['box']}, "
                f"expanded={expanded_box}, "
                f"crop_size={crop.size}, top={float(face_crop_top):.2f}, "
                f"bottom={float(face_crop_bottom):.2f}"
            )
        if emb is None:
            print(f"[BUBBLE_MATCH] 얼굴 {i} 임베딩 실패: {f['box']}")

    # ── 유사도 매트릭스 → 전역 최적 1:1 배정 ──
    # 후보를 개별 최고점 순서로 고르면 한 캐릭터가 다른 캐릭터의 유일한 얼굴을
    # 먼저 차지할 수 있다. 전체 조합에서 배정 인원 수와 유사도 합을 차례로 최대화한다.
    embedded_names = list(name_embs)
    similarities = []
    ranking_scores = []
    appearance_weight = max(0.0, float(appearance_weight or 0.0))
    ambiguity_margin = max(0.0, float(ambiguity_margin or 0.0))
    detection_confidence_weight = max(
        0.0, float(detection_confidence_weight or 0.0)
    )
    for n in embedded_names:
        row = []
        rank_row = []
        for fi, fe in enumerate(face_embs):
            sim = _cosine(name_embs[n], fe) if fe is not None else None
            row.append(sim)
            app_sim = appearance_similarity(
                name_appearances.get(n), face_appearances[fi]
            )
            if sim is None:
                combined = None
            elif app_sim is None or appearance_weight <= 0.0:
                combined = sim
            else:
                combined = (sim + appearance_weight * app_sim) / (1.0 + appearance_weight)
            face_conf = faces[fi].get("conf")
            confidence = max(0.0, float(face_conf or 0.0))
            ranking_score = (
                combined + detection_confidence_weight * confidence
                if combined is not None else None
            )
            rank_row.append(ranking_score)
            conf_text = f"{float(face_conf):.3f}" if face_conf is not None else "없음"
            sim_text = f"{sim:.3f}" if sim is not None else "계산불가"
            app_text = f"{app_sim:.3f}" if app_sim is not None else "계산불가"
            combined_text = f"{combined:.3f}" if combined is not None else "계산불가"
            rank_text = (
                f"{ranking_score:.3f}" if ranking_score is not None else "계산불가"
            )
            print(
                f"[BUBBLE_MATCH] 유사도: {n} ↔ 얼굴{fi} "
                f"(clip={sim_text}, appearance={app_text}, combined={combined_text}, "
                f"rank={rank_text}, "
                f"yolo_conf={conf_text}, box={faces[fi]['box']})"
            )
        similarities.append(row)
        ranking_scores.append(rank_row)

    optimal, assignment_steps = _sequential_overlap_assignment(
        similarities,
        ranking_scores,
        match_thres,
        expanded_face_boxes,
        ambiguity_margin=ambiguity_margin,
    )
    for step_number, step in enumerate(assignment_steps, start=1):
        gap = step["identity_gap"]
        gap_text = "단독" if not np.isfinite(gap) else f"{gap:.4f}"
        print(
            f"[BUBBLE_MATCH] 순차 확정 {step_number}: "
            f"{embedded_names[step['row']]} ↔ 얼굴{step['face']} "
            f"(identity_gap={gap_text}, rank={step['rank']:.3f}, "
            f"겹침 폐기={step['removed_faces']})"
        )
    name_to_face = {}
    for name_index, (fi, sim) in optimal.items():
        n = embedded_names[name_index]
        name_to_face[n] = (fi, sim)
        ranking_score = ranking_scores[name_index][fi]
        print(
            f"[BUBBLE_MATCH] 배정: {n} ↔ 얼굴{fi} "
            f"(box={faces[fi]['box']}, clip={sim:.3f}, rank={ranking_score:.3f})"
        )

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
            item = {"segment": s, "face_box": None, "char_name": sp, "sim": None}
            if sp and str(sp).casefold() in {
                str(name).casefold() for name in missing_project_names
            }:
                item["unmatched_reason"] = _MISSING_PROJECT_CHARACTER
            results.append(item)
    return results
