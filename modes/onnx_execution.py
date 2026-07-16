"""말풍선 모드 ONNX Runtime 장치와 CPU 스레드 정책.

설치된 Execution Provider와 현재 시스템의 논리 프로세서 수를 런타임에 읽는다.
각 추론 모듈은 이 파일의 정규화/세션 생성 함수를 공유해 같은 설정을 적용한다.
"""

import os
import traceback


AUTO_DEVICE = "auto"
CPU_DEVICE = "cpu"


def installed_providers():
    """현재 onnxruntime 빌드가 노출하는 provider 집합을 반환한다."""
    try:
        import onnxruntime as ort

        return set(ort.get_available_providers())
    except Exception as e:
        print(f"[ONNX_EXECUTION] provider 조회 실패: {e}")
        traceback.print_exc()
        return {"CPUExecutionProvider"}


def auto_device_key():
    """자동 장치 우선순위: CUDA > DirectML > CPU."""
    providers = installed_providers()
    if "CUDAExecutionProvider" in providers:
        return "cuda0"
    if "DmlExecutionProvider" in providers:
        return "dml0"
    return CPU_DEVICE


def list_devices():
    """UI 드롭다운에 표시할 설치 환경 기반 장치 목록."""
    devices = [
        {"key": AUTO_DEVICE, "label": "자동 (권장)", "provider": AUTO_DEVICE},
        {"key": CPU_DEVICE, "label": "CPU", "provider": "CPUExecutionProvider"},
    ]
    providers = installed_providers()
    if "CUDAExecutionProvider" in providers:
        devices.append({
            "key": "cuda0",
            "label": "CUDA · GPU (NVIDIA)",
            "provider": "CUDAExecutionProvider",
        })
    if "DmlExecutionProvider" in providers:
        devices.append({
            "key": "dml0",
            "label": "DirectML · GPU (Windows)",
            "provider": "DmlExecutionProvider",
        })
    return devices


def logical_cpu_count():
    """최소 1인 논리 프로세서 수."""
    count = os.cpu_count()
    if not count or count < 1:
        print(f"[ONNX_EXECUTION] 논리 프로세서 수 조회 실패({count!r}), 1 사용")
        return 1
    return int(count)


def list_cpu_thread_options():
    """자동과 1..논리 프로세서 수를 모두 노출한다."""
    count = logical_cpu_count()
    options = [{"value": 0, "label": "자동 (ONNX Runtime)"}]
    options.extend(
        {"value": threads, "label": f"{threads} 스레드"}
        for threads in range(1, count + 1)
    )
    return options


def normalize_device_key(device_key):
    """사용 불가능하거나 잘못된 장치는 자동으로 정규화한다."""
    key = str(device_key or AUTO_DEVICE).strip().lower() or AUTO_DEVICE
    valid = {item["key"] for item in list_devices()}
    if key not in valid:
        print(f"[ONNX_EXECUTION] 사용할 수 없는 장치({device_key!r}), 자동 사용")
        return AUTO_DEVICE
    return key


def normalize_cpu_threads(value):
    """0(자동) 또는 현재 환경의 1..논리 프로세서 수로 제한한다."""
    try:
        threads = int(value or 0)
    except (TypeError, ValueError):
        print(f"[ONNX_EXECUTION] CPU 스레드 값 변환 실패({value!r}), 자동 사용")
        return 0
    maximum = logical_cpu_count()
    if threads < 0:
        print(f"[ONNX_EXECUTION] 음수 CPU 스레드({threads}), 자동 사용")
        return 0
    if threads > maximum:
        print(
            f"[ONNX_EXECUTION] CPU 스레드 {threads}가 논리 프로세서 {maximum} 초과, "
            f"{maximum} 사용"
        )
        return maximum
    return threads


def resolved_device_key(device_key):
    key = normalize_device_key(device_key)
    return auto_device_key() if key == AUTO_DEVICE else key


def providers_for(device_key):
    """정규화된 장치 키에 대응하는 ORT provider 설정."""
    key = resolved_device_key(device_key)
    if key == CPU_DEVICE:
        return ["CPUExecutionProvider"]
    if key.startswith("cuda"):
        try:
            device_id = int(key[4:] or "0")
        except ValueError:
            print(f"[ONNX_EXECUTION] CUDA 장치 번호 변환 실패({key!r}), device_id=0 사용")
            device_id = 0
        return [("CUDAExecutionProvider", {"device_id": device_id})]
    if key.startswith("dml"):
        try:
            device_id = int(key[3:] or "0")
        except ValueError:
            print(f"[ONNX_EXECUTION] DirectML 장치 번호 변환 실패({key!r}), device_id=0 사용")
            device_id = 0
        return [("DmlExecutionProvider", {"device_id": device_id})]
    print(f"[ONNX_EXECUTION] 알 수 없는 장치 키({key!r}), CPU 사용")
    return ["CPUExecutionProvider"]


def session_cache_key(model_path, device_key=AUTO_DEVICE, cpu_threads=0):
    """자동 장치의 현재 해석 결과까지 포함한 안정적인 세션 캐시 키."""
    return (
        os.path.abspath(str(model_path)),
        resolved_device_key(device_key),
        normalize_cpu_threads(cpu_threads),
    )


def session_uses_gpu(session):
    """실제 세션 provider 목록에 CUDA 또는 DirectML이 있는지 확인한다."""
    try:
        providers = set(session.get_providers())
    except Exception as e:
        print(f"[ONNX_EXECUTION] 활성 provider 조회 실패: {e}")
        traceback.print_exc()
        return False
    return bool({"CUDAExecutionProvider", "DmlExecutionProvider"} & providers)


def cache_session(cache, key, session, *, log_prefix="ONNX", max_entries=4):
    """설정 탐색으로 모델 세션이 무제한 누적되지 않도록 LRU에 가깝게 보관한다."""
    if key in cache:
        cache.pop(key, None)
    cache[key] = session
    limit = max(1, int(max_entries))
    while len(cache) > limit:
        old_key = next(iter(cache))
        cache.pop(old_key, None)
        print(
            f"[{log_prefix}] 오래된 ONNX 세션 캐시 해제: "
            f"device={old_key[1]}, cpu_threads={old_key[2] or 'auto'}"
        )
    return session


def _session_options(device_key, cpu_threads, *, graph_optimization=True, log_severity=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    threads = normalize_cpu_threads(cpu_threads)
    options.intra_op_num_threads = threads
    # 현재 말풍선 모델들은 순차 그래프이며, 노드 내부 병렬화가 주된 가속 경로다.
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if graph_optimization:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if log_severity is not None:
        options.log_severity_level = int(log_severity)
    # DirectML은 메모리 패턴을 지원하지 않으며 순차 실행을 요구한다.
    if resolved_device_key(device_key).startswith("dml"):
        options.enable_mem_pattern = False
    return options


def create_session(
    model_path,
    *,
    device_key=AUTO_DEVICE,
    cpu_threads=0,
    log_prefix="ONNX",
    graph_optimization=True,
    log_severity=None,
):
    """선택 장치로 세션을 만들고 실패하면 같은 스레드 설정의 CPU로 폴백한다.

    Returns:
        ``(session, active_device_key)``. CPU 생성도 실패하면 ``(None, None)``.
    """
    import onnxruntime as ort

    requested = normalize_device_key(device_key)
    target = resolved_device_key(requested)
    threads = normalize_cpu_threads(cpu_threads)
    try:
        options = _session_options(
            target,
            threads,
            graph_optimization=graph_optimization,
            log_severity=log_severity,
        )
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers_for(target),
        )
        print(
            f"[{log_prefix}] ONNX 세션 생성: requested={requested}, resolved={target}, "
            f"cpu_threads={'auto' if threads == 0 else threads}, "
            f"providers={session.get_providers()}"
        )
        return session, target
    except Exception as e:
        print(
            f"[{log_prefix}] ONNX 세션 생성 실패: device={target}, "
            f"cpu_threads={'auto' if threads == 0 else threads}, error={e}"
        )
        traceback.print_exc()
        if target == CPU_DEVICE:
            return None, None

    print(f"[{log_prefix}] 선택 장치 실패로 CPU 폴백 시도")
    try:
        options = _session_options(
            CPU_DEVICE,
            threads,
            graph_optimization=graph_optimization,
            log_severity=log_severity,
        )
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        print(
            f"[{log_prefix}] CPU 폴백 세션 생성 완료: "
            f"cpu_threads={'auto' if threads == 0 else threads}, "
            f"providers={session.get_providers()}"
        )
        return session, CPU_DEVICE
    except Exception as e:
        print(f"[{log_prefix}] CPU 폴백 세션 생성 실패: {e}")
        traceback.print_exc()
        return None, None
