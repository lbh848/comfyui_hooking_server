"""
이미지 필터링(풀 축소) 모듈 — 그림체(스타일) LoRA 전용.

목적: 프로젝트의 전체 이미지 풀에서 학습에 쓸 대표 N장만 남기고 나머지를 정리.
두 전략:
  1) random  — 단순 무작위 N장 선택
  2) diverse — WD-vit-tagger-v3 의 10861차원 태그확률 벡터를 임베딩으로 사용하여
               임베딩 공간에서 가장 넓게 퍼진(farthest-point sampling) N장 선택.

임베딩은 기존 WD Tagger 모델(SmilingWolf/wd-vit-tagger-v3)의 원시 확률 출력을 재사용.
새 모델 다운로드/의존성 없음. CPU 고정. 배치 + 멀티스레드로 5000장 2분 타겟.

잡은 단일 슬롯(한 번에 하나만 실행). 진행률은 get_job_status() 로 폴링.
"""
import asyncio
import os
import random
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# WD Tagger 메타/전처리 재사용
from modes.wd_tagger_standalone import (
    WD_REPO, WD_MODEL_FILE, MODEL_SIZE, _download_if_needed, _model_cache_dir,
)

log_prefix = "[IMAGE_FILTER]"

# 배치 추론 크기. 벤치마크 결과 16 이상에서 수렴(~348ms/img). 메모리/속도 트레이드오프.
BATCH_SIZE = 16


class FilterCancelled(Exception):
    """사용자 중지 요청으로 임베딩이 중단되었을 때 발생."""


# 사용자 중지 요청 플래그 (스레드 간 안전한 Event)
_cancel = threading.Event()

# ─── ONNX 임베딩 세션 (lazy singleton) ──────────────────────────
_session = None


def _get_session():
    """임베딩 전용 ONNX 세션 반환. CPU 스레드를 최대한 활용하도록 튜닝."""
    global _session
    if _session is not None:
        return _session, _session.get_inputs()[0].name
    import onnxruntime as ort
    model_path = _download_if_needed(WD_REPO, WD_MODEL_FILE, _model_cache_dir)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # 전체 코어에서 2개 남기고 사용 (시스템 응답성 확보). 최소 1.
    cpu = max(1, (os.cpu_count() or 4) - 2)
    opts.intra_op_num_threads = cpu
    opts.inter_op_num_threads = 1
    _session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    print(f"{log_prefix} ONNX 임베딩 세션 준비 완료 "
          f"(intra_threads={cpu}, total_cpu={os.cpu_count()}, 2 cores reserved)")
    return _session, _session.get_inputs()[0].name


# ─── 전처리 ──────────────────────────────────────────────────────

def _preprocess_one_safe(path: str):
    """단일 이미지 -> (448,448,3) float32 (NHWC, raw 0~255, 정규화 없음 - v3 규격).
    실패 시 None 반환 (조용한 스킵 금지: 호출처에서 로깅)."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        max_dim = max(img.width, img.height)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, (0, 0))
        padded = padded.resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)
        return np.asarray(padded, dtype=np.float32)
    except Exception as e:
        print(f"{log_prefix} 전처리 실패 {path}: {type(e).__name__}: {e}")
        return None


# ─── 임베딩 ──────────────────────────────────────────────────────

def embed_images(paths, filenames, progress_cb):
    """이미지 경로 리스트 -> 임베딩 행렬. 청크(배치) 단위 추론.
    progress_cb(done_count) 를 매 배치마다 호출.
    반환: (valid_filenames, emb ndarray (N,D), failed_filenames)
    """
    session, input_name = _get_session()
    n = len(paths)
    valid_files = []
    vecs = []
    failed = []
    # 전처리 스레드도 코어 2개 남기고 사용. 단일 ONNX 추론 스레드(intra_threads)와
    # 겹쳐도 되지만, 과도한 스레드 경합을 막기 위해 동일한 상한을 적용.
    num_workers = max(1, min((os.cpu_count() or 4) - 2, 8))
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            chunk_paths = paths[start:end]
            chunk_fns = filenames[start:end]
            arrs = list(ex.map(_preprocess_one_safe, chunk_paths))

            batch_arr = []
            batch_local_idx = []
            for i, arr in enumerate(arrs):
                if arr is None:
                    failed.append(chunk_fns[i])
                    print(f"{log_prefix} 디코드 실패로 스킵: {chunk_fns[i]}")
                else:
                    batch_arr.append(arr)
                    batch_local_idx.append(i)

            if batch_arr:
                batch = np.ascontiguousarray(np.stack(batch_arr, axis=0))  # (B,448,448,3)
                preds = session.run(None, {input_name: batch})[0]  # (B, D) 확률
                for j, local_i in enumerate(batch_local_idx):
                    valid_files.append(chunk_fns[local_i])
                    vecs.append(preds[j].astype(np.float32))

            if progress_cb:
                progress_cb(end)

            # 사용자 중지 요청 확인 (배치 경계에서만 끊김 — 단일 ONNX 추론은 강제 kill 불가)
            if _cancel.is_set():
                print(f"{log_prefix} 사용자 중지 요청 감지 - 임베딩 중단 "
                      f"(processed={len(valid_files)}/{n})")
                raise FilterCancelled()

    if vecs:
        emb = np.vstack(vecs).astype(np.float32)
    else:
        emb = np.zeros((0, 1), dtype=np.float32)
        print(f"{log_prefix} 경고: 유효 임베딩 0건 (전체 {n}장 전부 실패?)")

    dt = time.time() - t0
    avg = dt * 1000 / max(len(valid_files), 1)
    print(f"{log_prefix} 임베딩 완료: {len(valid_files)}/{n} in {dt:.1f}s, "
          f"avg={avg:.0f}ms/img, failed={len(failed)}")
    return valid_files, emb, failed


# ─── 선택 알고리즘 ───────────────────────────────────────────────

def select_random(filenames, k):
    """무작위 k개 선택. k 가 전체보다 크면 전체 반환."""
    n = len(filenames)
    if k >= n:
        return list(filenames)
    idx = random.sample(range(n), k)
    return [filenames[i] for i in idx]


def farthest_point_sampling(emb: np.ndarray, k: int):
    """탐욕 최장거리 샘플링(max-min cosine distance).
    임베딩 공간에서 가장 넓게 퍼진 k개의 인덱스 반환.
    seed = 중심(centroid)으로부터 가장 먼 점. 이후 매 스텝
    '이미 선택된 점들과의 최대 유사도'가 가장 작은(=가장 다양한) 점을 선택.
    """
    n = emb.shape[0]
    if k >= n:
        return list(range(n))

    # L2 정규화 → 코사인 유사도 = 내적
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    E = emb / norms  # (N, D)

    # seed: centroid 로부터 가장 먼(가장 다양한) 점
    centroid = E.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm > 0:
        sims_centroid = E @ (centroid / cnorm)
        seed = int(np.argmin(sims_centroid))
    else:
        seed = 0

    picked = np.zeros(n, dtype=bool)
    picked[seed] = True
    selected = [seed]
    best_sim = E @ E[seed]  # (N,) 각 점의 '가장 가까운 선택점'과의 유사도

    for _ in range(1, k):
        # 선택된 점은 inf 로 마스킹 후 argmin
        cand = np.where(picked, np.inf, best_sim)
        nxt = int(np.argmin(cand))
        picked[nxt] = True
        selected.append(nxt)
        best_sim = np.maximum(best_sim, E @ E[nxt])

    return selected


# ─── 잡 관리 (단일 슬롯) ─────────────────────────────────────────

_job = {}


def get_job_status() -> dict:
    if not _job:
        return {"status": "idle", "stage": "", "total": 0, "done": 0}
    return dict(_job)


def start_filter_job(project: str, mode: str, count: int) -> dict:
    """필터링 잡 시작. 단일 슬롯. 성공 시 {success, job_id}."""
    if _job.get("status") == "running":
        return {"success": False,
                "error": "이미 필터링 작업이 실행 중입니다. 완료 후 다시 시도하세요."}

    from modes import style_lora_mode
    detail = style_lora_mode.get_project_detail(project)
    if not detail.get("success"):
        return {"success": False, "error": detail.get("error", "프로젝트 조회 실패")}
    all_files = detail["data"].get("images", [])
    total = len(all_files)
    if total == 0:
        return {"success": False, "error": "프로젝트에 이미지가 없습니다"}

    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return {"success": False, "error": f"개수는 1 이상의 정수여야 합니다 (입력: {count!r})"}
    if count > total:
        return {"success": False,
                "error": f"선택 개수({count})가 전체 이미지 수({total})를 초과합니다"}
    if mode not in ("random", "diverse"):
        return {"success": False, "error": f"알 수 없는 모드: {mode!r}"}

    job_id = f"filter_{uuid.uuid4().hex[:8]}"
    _job.clear()
    _cancel.clear()  # 이전 중지 요청 플래그 초기화
    _job.update({
        "id": job_id,
        "status": "running",
        "stage": "init",
        "total": total,
        "done": 0,
        "mode": mode,
        "count": count,
        "selected": [],
        "to_delete": [],
        "error": None,
        "project": project,
    })

    print(f"{log_prefix} 잡 시작: project={project} mode={mode} "
          f"count={count} total={total}")
    asyncio.create_task(_run_job(job_id, project, mode, count, all_files))
    return {"success": True, "job_id": job_id}


async def _run_job(job_id, project, mode, count, all_files):
    """백그라운드 잡 본체. 무거운 추론/선택은 스레드풀로 오프로드."""
    from modes import style_lora_mode
    try:
        if mode == "random":
            _job["stage"] = "selecting"
            sel = select_random(all_files, count)
            sel_set = set(sel)
            _job["selected"] = sel
            _job["to_delete"] = [f for f in all_files if f not in sel_set]
            _job["done"] = len(all_files)
            _job["status"] = "done"
            print(f"{log_prefix} random 완료: keep={len(sel)} "
                  f"del={len(_job['to_delete'])}")
            return

        # diverse: 임베딩 -> farthest-point sampling
        _job["stage"] = "embedding"
        loop = asyncio.get_event_loop()
        paths = [style_lora_mode.get_image_path(project, f) for f in all_files]

        def _cb(done):
            _job["done"] = done

        valid_files, emb, failed = await loop.run_in_executor(
            None, embed_images, paths, all_files, _cb
        )

        if len(valid_files) == 0:
            raise RuntimeError("임베딩 가능한 이미지가 한 장도 없습니다")

        # count 가 유효 이미지 수보다 크면 유효 이미지 전체 선택
        k = min(count, len(valid_files))
        _job["stage"] = "selecting"
        local_sel = await loop.run_in_executor(
            None, farthest_point_sampling, emb, k
        )
        sel = [valid_files[i] for i in local_sel]
        sel_set = set(sel)
        # 삭제 대상 = 전체에서 선택된 것 외 전부 (failed 포함)
        to_delete = [f for f in all_files if f not in sel_set]
        _job["selected"] = sel
        _job["to_delete"] = to_delete
        _job["done"] = len(all_files)
        _job["status"] = "done"
        print(f"{log_prefix} diverse 완료: keep={len(sel)} "
              f"del={len(to_delete)} (failed_img={len(failed)})")

    except FilterCancelled:
        print(f"{log_prefix} 잡 중지됨 (사용자 요청)")
        _job["status"] = "cancelled"
        _job["stage"] = "cancelled"
        _job["error"] = "사용자가 중지했습니다"

    except Exception as e:
        print(f"{log_prefix} 잡 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        _job["status"] = "error"
        _job["error"] = f"{type(e).__name__}: {e}"


def cancel_filter_job() -> dict:
    """실행 중인 필터링 잡에 중지 요청. 실제 중단은 현재 배치가 끝난 시점(수 초 내)."""
    if _job.get("status") == "running":
        _cancel.set()
        print(f"{log_prefix} 중지 요청 수신 (현재 배치 완료 후 중단)")
        return {"success": True, "message": "중지 요청됨 (현재 배치 완료 후 중단)"}
    return {"success": False, "error": "실행 중인 필터링 작업이 없습니다"}
