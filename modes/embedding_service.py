"""
embedding_service - 텍스트 임베딩 서비스

에셋툴 분석 결과에서 감정/상황 키워드와 프리셋 간의
의미적 유사도를 계산하기 위한 임베딩 기반 매칭 제공.

지원 프로바이더:
  - voyage: Voyage AI (기본, POST https://api.voyageai.com/v1/embeddings)
  - custom: OpenAI 호환 임베딩 API (임의 URL)

설정(config.json):
  embedding_provider: "voyage" | "custom"
  embedding_url: 임베딩 API 엔드포인트 URL
  embedding_api_key: API 키
  embedding_model: 모델명

프로필(asset_data/embedding_profile_map.json):
  clean_profiles: 정제 규칙 프로필 사전 {이름: [step, ...]}
  active_preset_profile: 활성 프리셋 프로필 이름
  active_tag_profile: 활성 태그 프로필 이름
"""

import json
import math
import os
import re
import time
import traceback
from typing import Optional

import httpx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
EMBEDDING_CACHE_FILE = os.path.join(BASE_DIR, "asset_data", "embedding_cache.json")
PROFILE_MAP_FILE = os.path.join(BASE_DIR, "asset_data", "embedding_profile_map.json")

DEFAULT_CLEAN_PROFILES = {
    "기본 (프리셋)": [
        {"action": "split_by", "separator": "/", "take": -1},
        {"action": "strip_suffix", "pattern": "_v\\d+$"},
        {"action": "replace", "from": "_", "to": " "},
    ],
    "기본 (태그)": [
        {"action": "strip_suffix", "pattern": "\\.(webp|png|jpg|jpeg|gif|bmp)$"},
        {"action": "split_by", "separator": "_", "take": "1:"},
        {"action": "replace", "from": "_", "to": " "},
    ],
}

_current_config = {
    "embedding_provider": "voyage",
    "embedding_url": "https://api.voyageai.com/v1/embeddings",
    "embedding_api_key": "",
    "embedding_model": "voyage-4-large",
    "clean_profiles": dict(DEFAULT_CLEAN_PROFILES),
    "active_preset_profile": "기본 (프리셋)",
    "active_tag_profile": "기본 (태그)",
}

_embedding_cache: dict[str, list[float]] = {}

_local_cache: Optional[dict] = None

DEFAULT_URLS = {
    "voyage": "https://api.voyageai.com/v1/embeddings",
}


def _parse_take(take_value) -> object:
    if isinstance(take_value, int):
        return take_value
    if isinstance(take_value, str):
        if take_value.endswith(":"):
            try:
                return (int(take_value[:-1]), None)
            except ValueError:
                return take_value
        if take_value.startswith(":"):
            try:
                return (None, int(take_value[1:]))
            except ValueError:
                return take_value
        if ":" in take_value:
            parts = take_value.split(":", 1)
            try:
                return (int(parts[0]) if parts[0] else None,
                        int(parts[1]) if parts[1] else None)
            except ValueError:
                return take_value
    return take_value


def _apply_take(parts: list[str], take_value) -> str:
    if isinstance(take_value, int):
        if -len(parts) <= take_value < len(parts):
            return parts[take_value]
        return parts[-1] if take_value < 0 else parts[0]

    if isinstance(take_value, tuple):
        start, end = take_value
        if start is not None and start < 0:
            start = len(parts) + start
        if end is not None and end < 0:
            end = len(parts) + end
        selected = parts[start:end]
        return "_".join(selected)

    return parts[take_value] if isinstance(take_value, int) and -len(parts) <= take_value < len(parts) else parts[-1]


def clean_name_by_steps(name: str, steps: list[dict]) -> str:
    result = name
    for step in steps:
        action = step.get("action", "")
        if action == "split_by":
            sep = step.get("separator", "_")
            take = step.get("take", -1)
            parts = result.split(sep)
            if not parts:
                continue
            take_parsed = _parse_take(take)
            result = _apply_take(parts, take_parsed)
        elif action == "strip_suffix":
            pattern = step.get("pattern", "")
            if pattern:
                result = re.sub(pattern, "", result)
        elif action == "strip_prefix":
            pattern = step.get("pattern", "")
            if pattern:
                result = re.sub(f"^{pattern}", "", result)
        elif action == "replace":
            from_str = step.get("from", "")
            to_str = step.get("to", " ")
            result = result.replace(from_str, to_str)
        elif action == "strip":
            result = result.strip()
        elif action == "lower":
            result = result.lower()
    return result.strip()


def _get_active_steps(profile_type: str) -> list[dict]:
    if profile_type == "preset":
        profile_name = _current_config.get("active_preset_profile", "")
    else:
        profile_name = _current_config.get("active_tag_profile", "")
    profiles = _current_config.get("clean_profiles", {})
    if profile_name and profile_name in profiles:
        return profiles[profile_name]
    fallback_key = "기본 (프리셋)" if profile_type == "preset" else "기본 (태그)"
    return profiles.get(fallback_key, DEFAULT_CLEAN_PROFILES.get(fallback_key, []))


def _signature_for_config() -> dict:
    profiles = _current_config.get("clean_profiles", {})
    return {
        "provider": _current_config.get("embedding_provider", "voyage"),
        "model": _current_config.get("embedding_model", "voyage-4-large"),
        "url": _current_config.get("embedding_url", ""),
        "clean_profiles": profiles,
        "active_preset_profile": _current_config.get("active_preset_profile", ""),
        "active_tag_profile": _current_config.get("active_tag_profile", ""),
    }


def _load_local_cache() -> dict:
    global _local_cache
    if _local_cache is not None:
        return _local_cache
    if os.path.isfile(EMBEDDING_CACHE_FILE):
        try:
            with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                _local_cache = json.load(f)
            _log(f"로컬 캐시 로드: {len(_local_cache.get('embeddings', {}))}개")
        except Exception as e:
            _log(f"로컬 캐시 로드 실패: {e}")
            _local_cache = {"signature": {}, "embeddings": {}}
    else:
        _local_cache = {"signature": {}, "embeddings": {}}
    return _local_cache


def _save_local_cache():
    global _local_cache
    if _local_cache is None:
        return
    cache_dir = os.path.dirname(EMBEDDING_CACHE_FILE)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_local_cache, f, ensure_ascii=False)
        _log(f"로컬 캐시 저장: {len(_local_cache.get('embeddings', {}))}개")
    except Exception as e:
        _log(f"로컬 캐시 저장 실패: {e}")


def _is_cache_valid() -> bool:
    cache = _load_local_cache()
    saved_sig = cache.get("signature", {})
    current_sig = _signature_for_config()
    return saved_sig == current_sig


def _invalidate_local_cache():
    global _local_cache
    _local_cache = {"signature": {}, "embeddings": {}}
    if os.path.isfile(EMBEDDING_CACHE_FILE):
        try:
            os.remove(EMBEDDING_CACHE_FILE)
            _log("로컬 캐시 파일 삭제 (설정 변경으로 무효화)")
        except Exception as e:
            _log(f"로컬 캐시 파일 삭제 실패: {e}")


def _migrate_legacy_config():
    global _current_config
    if "preset_clean_steps" in _current_config and "clean_profiles" not in _current_config:
        _log("레거시 preset_clean_steps/tag_clean_steps → clean_profiles 마이그레이션")
        old_preset = _current_config.pop("preset_clean_steps", [])
        old_tag = _current_config.pop("tag_clean_steps", [])
        _current_config["clean_profiles"] = {
            "기본 (프리셋)": old_preset if old_preset else DEFAULT_CLEAN_PROFILES["기본 (프리셋)"],
            "기본 (태그)": old_tag if old_tag else DEFAULT_CLEAN_PROFILES["기본 (태그)"],
        }
        _current_config["active_preset_profile"] = "기본 (프리셋)"
        _current_config["active_tag_profile"] = "기본 (태그)"


def update_config(config: dict):
    global _current_config
    _profile_keys = {"clean_profiles", "active_preset_profile", "active_tag_profile"}
    for key, value in config.items():
        if key in _profile_keys:
            continue
        if key in _current_config:
            _current_config[key] = value
    _migrate_legacy_config()
    provider = _current_config.get("embedding_provider", "voyage")
    url = _current_config.get("embedding_url", "")
    if not url and provider in DEFAULT_URLS:
        _current_config["embedding_url"] = DEFAULT_URLS[provider]
    old_sig = _signature_for_config()
    new_sig = _signature_for_config()
    if old_sig != new_sig:
        _log("설정 변경으로 임베딩 캐시 무효화")
        _invalidate_local_cache()
    _log(f"설정 업데이트: provider={_current_config['embedding_provider']}, "
         f"url={_current_config['embedding_url']}, "
         f"model={_current_config['embedding_model']}, "
         f"api_key={'(empty)' if not _current_config.get('embedding_api_key') else _current_config['embedding_api_key']}")


def get_config() -> dict:
    return _current_config.copy()


def get_clean_steps(key: str) -> list[dict]:
    if key == "preset_clean_steps" or key == "preset":
        return _get_active_steps("preset")
    elif key == "tag_clean_steps" or key == "tag":
        return _get_active_steps("tag")
    steps = _current_config.get(key)
    if steps is None:
        steps = DEFAULT_CLEAN_PROFILES.get("기본 (프리셋)", [])
    return steps


def list_profiles() -> dict:
    return dict(_current_config.get("clean_profiles", {}))


def _save_profiles_to_file():
    profiles = _current_config.get("clean_profiles", {})
    profile_map = {}
    if os.path.isfile(PROFILE_MAP_FILE):
        try:
            with open(PROFILE_MAP_FILE, "r", encoding="utf-8") as f:
                profile_map = json.load(f)
        except Exception:
            profile_map = {}
    profile_map["clean_profiles"] = profiles
    profile_map["active_preset_profile"] = _current_config.get("active_preset_profile", "")
    profile_map["active_tag_profile"] = _current_config.get("active_tag_profile", "")
    os.makedirs(os.path.dirname(PROFILE_MAP_FILE), exist_ok=True)
    try:
        with open(PROFILE_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(profile_map, f, indent=2, ensure_ascii=False)
        _log(f"프로필 파일 저장 완료 ({len(profiles)}개)")
    except Exception as e:
        _log(f"프로필 파일 저장 실패: {e}")


def _load_profiles_from_file():
    global _current_config
    if not os.path.isfile(PROFILE_MAP_FILE):
        return
    try:
        with open(PROFILE_MAP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "clean_profiles" in data:
            _current_config["clean_profiles"] = data["clean_profiles"]
        if "active_preset_profile" in data:
            _current_config["active_preset_profile"] = data["active_preset_profile"]
        if "active_tag_profile" in data:
            _current_config["active_tag_profile"] = data["active_tag_profile"]
        _log(f"프로필 파일 로드 완료 ({len(_current_config.get('clean_profiles', {}))}개)")
    except Exception as e:
        _log(f"프로필 파일 로드 실패: {e}")


def save_profile(name: str, steps: list[dict]) -> dict:
    global _current_config
    if not name or not name.strip():
        return {"success": False, "error": "프로필 이름이 필요합니다"}
    name = name.strip()
    profiles = _current_config.setdefault("clean_profiles", {})
    profiles[name] = steps
    _save_profiles_to_file()
    _log(f"프로필 저장: '{name}' ({len(steps)}단계)")
    return {"success": True, "name": name}


def delete_profile(name: str) -> dict:
    global _current_config
    profiles = _current_config.get("clean_profiles", {})
    if name not in profiles:
        return {"success": False, "error": f"프로필 '{name}'을 찾을 수 없습니다"}
    if len(profiles) <= 1:
        return {"success": False, "error": "마지막 프로필은 삭제할 수 없습니다"}
    if _current_config.get("active_preset_profile") == name:
        fallback = next((k for k in profiles if k != name), "기본 (프리셋)")
        _current_config["active_preset_profile"] = fallback
    if _current_config.get("active_tag_profile") == name:
        fallback = next((k for k in profiles if k != name), "기본 (태그)")
        _current_config["active_tag_profile"] = fallback
    del profiles[name]
    _save_profiles_to_file()
    _log(f"프로필 삭제: '{name}'")
    return {"success": True, "deleted": name}


def set_active_profile(profile_type: str, name: str) -> dict:
    global _current_config
    profiles = _current_config.get("clean_profiles", {})
    if name not in profiles:
        return {"success": False, "error": f"프로필 '{name}'을 찾을 수 없습니다"}
    if profile_type == "preset":
        _current_config["active_preset_profile"] = name
    elif profile_type == "tag":
        _current_config["active_tag_profile"] = name
    else:
        return {"success": False, "error": f"알 수 없는 프로필 타입: {profile_type}"}
    _save_profiles_to_file()
    _log(f"활성 프로필 변경 ({profile_type}): '{name}'")
    return {"success": True, "profile_type": profile_type, "active": name}


def get_preset_profile_map() -> dict:
    data = {}
    if os.path.isfile(PROFILE_MAP_FILE):
        try:
            with open(PROFILE_MAP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _log(f"프로필 맵 로드 실패: {e}")
    # 프로필 맵만 반환 (clean_profiles 등 제외)
    return {k: v for k, v in data.items() if k not in ("clean_profiles", "active_preset_profile", "active_tag_profile")}


def set_preset_profile_map(profile_map: dict) -> dict:
    os.makedirs(os.path.dirname(PROFILE_MAP_FILE), exist_ok=True)
    # 기존 파일에서 프로필 정보 보존
    existing = {}
    if os.path.isfile(PROFILE_MAP_FILE):
        try:
            with open(PROFILE_MAP_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(profile_map)
    try:
        with open(PROFILE_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        _log(f"프로필 맵 저장: {len(profile_map)}개 항목")
        return {"success": True, "count": len(profile_map)}
    except Exception as e:
        _log(f"프로필 맵 저장 실패: {e}")
        return {"success": False, "error": str(e)}


def clean_name_with_profile(name: str, profile_name: str = "") -> str:
    if profile_name:
        profiles = _current_config.get("clean_profiles", {})
        steps = profiles.get(profile_name, [])
        if steps:
            return clean_name_by_steps(name, steps)
    steps = _get_active_steps("preset")
    return clean_name_by_steps(name, steps)


def _log(message: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "embedding_service.log")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    print(f"[EMBED] {message}")


def _build_request(texts: list[str], input_type: str) -> tuple[dict, dict]:
    provider = _current_config.get("embedding_provider", "voyage")
    model = _current_config.get("embedding_model", "voyage-4-large")
    api_key = _current_config.get("embedding_api_key", "")
    url = _current_config.get("embedding_url", DEFAULT_URLS.get(provider, ""))

    if not url:
        raise ValueError("임베딩 URL이 설정되지 않았습니다")

    if not api_key:
        raise ValueError("임베딩 API 키가 설정되지 않았습니다")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if provider == "voyage":
        body = {
            "input": texts,
            "model": model,
            "input_type": input_type,
        }
    else:
        body = {
            "input": texts,
            "model": model,
        }

    return url, headers, body


async def get_embeddings(texts: list[str], input_type: str = "query") -> Optional[list[list[float]]]:
    api_key = _current_config.get("embedding_api_key", "")
    if not api_key:
        _log("API 키가 설정되지 않음")
        return None

    model = _current_config.get("embedding_model", "voyage-4-large")
    provider = _current_config.get("embedding_provider", "voyage")

    try:
        url, headers, body = _build_request(texts, input_type)
    except ValueError as e:
        _log(str(e))
        return None

    _log(f"임베딩 요청: {len(texts)}개 텍스트, input_type={input_type}, "
         f"provider={provider}, model={model}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=body, headers=headers)
            _log(f"임베딩 응답: status={response.status_code}")

            if response.status_code == 200:
                result = response.json()
                data = result.get("data", [])
                embeddings = [item["embedding"] for item in data]
                usage = result.get("usage", {})
                _log(f"임베딩 성공: {len(embeddings)}개, "
                     f"total_tokens={usage.get('total_tokens', 'N/A')}")
                return embeddings
            else:
                error_text = response.text[:500]
                _log(f"임베딩 실패: {response.status_code} - {error_text}")
                return None
    except httpx.TimeoutException:
        _log("임베딩 타임아웃")
        return None
    except Exception as e:
        _log(f"임베딩 예외: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def get_embedding_cached(text: str, input_type: str = "query") -> Optional[list[float]]:
    cache_key = f"{input_type}:{text}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    cache = _load_local_cache()
    if _is_cache_valid():
        emb = cache.get("embeddings", {}).get(cache_key)
        if emb is not None:
            _embedding_cache[cache_key] = emb
            return emb

    results = await get_embeddings([text], input_type=input_type)
    if results and len(results) > 0:
        _embedding_cache[cache_key] = results[0]
        return results[0]
    return None


async def batch_get_embeddings(texts: list[str], input_type: str = "document",
                                batch_size: int = 64) -> Optional[list[list[float]]]:
    all_embeddings: list[Optional[list[float]]] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    cache = _load_local_cache()
    local_has_valid_sig = _is_cache_valid()
    local_embs = cache.get("embeddings", {})

    for i, text in enumerate(texts):
        cache_key = f"{input_type}:{text}"
        if cache_key in _embedding_cache:
            all_embeddings[i] = _embedding_cache[cache_key]
        elif local_has_valid_sig and cache_key in local_embs:
            all_embeddings[i] = local_embs[cache_key]
            _embedding_cache[cache_key] = local_embs[cache_key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        _log(f"배치 임베딩: 전체 {len(texts)}개 중 캐시 미스 {len(uncached_texts)}개")

        for start in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[start:start + batch_size]
            batch_indices = uncached_indices[start:start + batch_size]

            results = await get_embeddings(batch_texts, input_type=input_type)
            if results is None:
                return None

            for j, (idx, embedding) in enumerate(zip(batch_indices, results)):
                all_embeddings[idx] = embedding
                cache_key = f"{input_type}:{uncached_texts[start + j]}"
                _embedding_cache[cache_key] = embedding

    if any(e is None for e in all_embeddings):
        _log("일부 임베딩이 누락됨")
        return None

    return all_embeddings


async def match_presets_by_names(
    tag_names: list[str],
    preset_names: list[str],
    tag_category: str = "expressions",
    top_n: int = 10,
    threshold: float = 0.3,
    tags_data: dict = None,
) -> list[dict]:
    try:
        api_key = _current_config.get("embedding_api_key", "")
        if not api_key:
            _log("API 키 없음 - 태그 기반 임베딩 매칭 스킵")
            return []

        if not tag_names or not preset_names:
            _log(f"태그 기반 임베딩 매칭 스킵: tag_names={len(tag_names) if tag_names else 0}, preset_names={len(preset_names) if preset_names else 0}")
            return []

        clean_tags = [_clean_tag_name(t) for t in tag_names]
        clean_tags = [c for c in clean_tags if c]
        if not clean_tags:
            _log(f"태그 기반 임베딩 매칭 스킵: 정제 후 태그 없음 (원본: {tag_names})")
            return []

        clean_presets = [_clean_preset_name(n, tags_data) for n in preset_names]
        unique_clean_tags = list(dict.fromkeys(clean_tags))
        unique_clean_presets = list(dict.fromkeys(clean_presets))
        unique_clean_presets = [c for c in unique_clean_presets if c]

        _log(f"태그 기반 임베딩 매칭 시작: 원본태그={len(tag_names)}, 정제태그={len(unique_clean_tags)}, 프리셋={len(unique_clean_presets)}")

        tag_embs = await batch_get_embeddings(unique_clean_tags, input_type="query")
        preset_embs = await batch_get_embeddings(unique_clean_presets, input_type="document")

        if tag_embs is None:
            _log("태그 기반 임베딩 매칭 실패: 태그 임베딩 결과가 None")
            return []
        if preset_embs is None:
            _log("태그 기반 임베딩 매칭 실패: 프리셋 임베딩 결과가 None")
            return []

        tag_emb_map = dict(zip(unique_clean_tags, tag_embs))
        preset_emb_map = dict(zip(unique_clean_presets, preset_embs))

        results = {}
        for tag_name, clean_tag in zip(tag_names, clean_tags):
            if not clean_tag or clean_tag not in tag_emb_map:
                continue
            tag_emb = tag_emb_map[clean_tag]
            for preset_name, clean_preset in zip(preset_names, clean_presets):
                if not clean_preset or clean_preset not in preset_emb_map:
                    continue
                sim = cosine_similarity(tag_emb, preset_emb_map[clean_preset])
                if sim >= threshold:
                    key = preset_name
                    if key not in results or sim > results[key]["similarity"]:
                        results[key] = {
                            "name": preset_name,
                            "clean_name": clean_preset,
                            "similarity": round(sim, 4),
                            "matched_tag": clean_tag,
                            "original_tag": tag_name,
                        }

        results_list = sorted(results.values(), key=lambda x: x["similarity"], reverse=True)
        _log(f"태그 기반 임베딩 매칭: {len(clean_tags)}개 태그 vs {len(preset_names)}개 프리셋, "
             f"결과 {len(results_list)}개 (threshold={threshold})")
    except Exception as e:
        _log(f"태그 기반 임베딩 매칭 예외: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []

    cache = _load_local_cache()
    cache["signature"] = _signature_for_config()
    cache["embeddings"].update({k: v for k, v in _embedding_cache.items()})
    _local_cache = cache
    _save_local_cache()

    return results_list[:top_n]


def _clean_preset_name(name: str, tags_data: dict = None) -> str:
    if tags_data:
        profile_map = tags_data.get("preset_profile_map", {})
        if name in profile_map:
            profile_name = profile_map[name]
            profiles = _current_config.get("clean_profiles", {})
            steps = profiles.get(profile_name, [])
            if steps:
                return clean_name_by_steps(name, steps)
    steps = _get_active_steps("preset")
    return clean_name_by_steps(name, steps)


def _clean_tag_name(name: str) -> str:
    steps = _get_active_steps("tag")
    return clean_name_by_steps(name, steps)


async def match_presets_by_query(
    query_text: str,
    preset_names: list[str],
    tag_category: str = "expressions",
    top_n: int = 5,
    threshold: float = 0.3,
    tags_data: dict = None,
) -> list[dict]:
    api_key = _current_config.get("embedding_api_key", "")
    if not api_key:
        _log("API 키 없음 - 임베딩 매칭 스킵")
        return []

    if not preset_names:
        return []

    clean_names = [_clean_preset_name(n, tags_data) for n in preset_names]
    unique_clean = list(dict.fromkeys(clean_names))
    unique_clean = [c for c in unique_clean if c]

    query_emb = await get_embedding_cached(query_text, input_type="query")
    if query_emb is None:
        return []

    preset_embs = await batch_get_embeddings(unique_clean, input_type="document")
    if preset_embs is None:
        return []

    clean_to_emb = dict(zip(unique_clean, preset_embs))

    results = []
    for preset_name, clean_name in zip(preset_names, clean_names):
        emb = clean_to_emb.get(clean_name)
        if emb is None:
            continue
        sim = cosine_similarity(query_emb, emb)
        if sim >= threshold:
            results.append({
                "name": preset_name,
                "clean_name": clean_name,
                "similarity": round(sim, 4),
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    _log(f"유사도 매칭 결과: query='{query_text}', 카테고리={tag_category}, "
         f"결과 {len(results)}/{len(preset_names)}개 (threshold={threshold})")
    return results[:top_n]


def preview_clean_names(names: list[str], steps: list[dict]) -> list[dict]:
    results = []
    for name in names:
        cleaned = clean_name_by_steps(name, steps)
        results.append({
            "original": name,
            "cleaned": cleaned,
        })
    return results


async def build_preset_embeddings(tags_data: dict, progress_callback=None, skip_cached: bool = False) -> dict:
    api_key = _current_config.get("embedding_api_key", "")
    if not api_key:
        _log("API 키 없음 - 임베딩 빌드 불가")
        return {"success": False, "error": "API 키가 설정되지 않았습니다"}

    all_names_to_embed: dict[str, tuple[str, str, str]] = {}
    name_map: dict[str, dict[str, str]] = {}

    categories = {
        "expressions": tags_data.get("expressions", {}),
        "appearances": tags_data.get("appearances", {}),
        "outfits": tags_data.get("outfits", {}),
        "composition_presets": tags_data.get("composition_presets", {}),
        "quality_presets": tags_data.get("quality_presets", {}),
        "negative_presets": tags_data.get("negative_presets", {}),
    }

    for cat_key, cat_data in categories.items():
        if not isinstance(cat_data, dict):
            continue
        name_map[cat_key] = {}
        for preset_name in cat_data:
            if not isinstance(cat_data[preset_name], list):
                continue
            clean = _clean_preset_name(preset_name, tags_data)
            name_map[cat_key][preset_name] = clean
            if clean and clean not in all_names_to_embed:
                all_names_to_embed[clean] = (cat_key, preset_name, "document")

    char_presets = tags_data.get("character_presets", {})
    if isinstance(char_presets, dict):
        name_map["character_presets"] = {}
        for char_name, char_data in char_presets.items():
            if not isinstance(char_data, dict):
                continue
            name_map["character_presets"][char_name] = char_name
            if char_name not in all_names_to_embed:
                all_names_to_embed[char_name] = ("character_presets", char_name, "document")

    unique_names = list(all_names_to_embed.keys())

    if skip_cached:
        cache = _load_local_cache()
        cached_embs = cache.get("embeddings", {})
        names_to_request = []
        for name in unique_names:
            cache_key = f"document:{name}"
            if cache_key in _embedding_cache or cache_key in cached_embs:
                continue
            names_to_request.append(name)
        _log(f"임베딩 빌드 시작 (캐시 스킵): 전체 {len(unique_names)}개 중 캐시 {len(unique_names) - len(names_to_request)}개 스킵, 신규 {len(names_to_request)}개")
        unique_names_for_api = names_to_request
    else:
        _log(f"임베딩 빌드 시작 (전체 재구축): {len(unique_names)}개 고유 이름")
        unique_names_for_api = unique_names

    total_count = len(unique_names)

    if progress_callback:
        await progress_callback(total_count - len(unique_names_for_api), total_count, "임베딩 계산 준비 중")

    batch_size = 64
    total_batches = (len(unique_names_for_api) + batch_size - 1) // batch_size
    embedded_count = total_count - len(unique_names_for_api)

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = unique_names_for_api[start:start + batch_size]
        results = await get_embeddings(batch, input_type="document")

        if results is None:
            _log(f"배치 {batch_idx + 1}/{total_batches} 임베딩 실패")
            if progress_callback:
                await progress_callback(embedded_count, total_count, f"배치 {batch_idx + 1} 실패")
            continue

        for name, emb in zip(batch, results):
            cache_key = f"document:{name}"
            _embedding_cache[cache_key] = emb

        embedded_count += len(batch)

        if progress_callback:
            await progress_callback(embedded_count, total_count,
                                    f"배치 {batch_idx + 1}/{total_batches} 완료")

    cache = _load_local_cache()
    cache["signature"] = _signature_for_config()
    cache["embeddings"].update({k: v for k, v in _embedding_cache.items()})
    _local_cache = cache
    _save_local_cache()

    skipped = total_count - len(unique_names_for_api)
    _log(f"임베딩 빌드 완료: 신규 {embedded_count - skipped if skip_cached else embedded_count}/{total_count}개"
         f"{f' (캐시 스킵 {skipped}개)' if skip_cached else ''}, "
         f"캐시 크기={len(cache['embeddings'])}개")

    return {
        "success": True,
        "total_names": total_count,
        "embedded_count": embedded_count,
        "skipped": skipped if skip_cached else 0,
        "name_map": name_map,
        "cache_size": len(cache["embeddings"]),
    }


def clear_cache():
    global _embedding_cache, _local_cache
    _embedding_cache = {}
    _local_cache = {"signature": {}, "embeddings": {}}
    _save_local_cache()
    _log("임베딩 캐시 전체 초기화")

_migrate_legacy_config()