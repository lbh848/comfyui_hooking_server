"""
Outfit Prompt Test Runner
서버 실행 없이 복장 추출 프롬프트를 테스트하고 결과를 검증하는 도구

사용법:
    python main.py [옵션]

옵션:
    --prompt FILE       테스트할 프롬프트 파일 (기본: haganai_extract_outfit_prompt.py)
    --real              workflow_backup의 실제 데이터 사용
    --show              입력 데이터와 전체 프롬프트 표시
    --filter NAME       캐릭터 이름으로 필터링
    --model MODEL       LLM 모델 지정 (기본: config.json 설정)
    --raw               LLM 호출 없이 프롬프트 빌드만 테스트
    --help              도움말
"""

import asyncio
import json
import sys
import os
import time
import re
import importlib.util
import io

# Windows cp949 유니코드 출력 문제 해결
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─── Path Setup ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from modes.llm_service import callLLM, callLLM2, update_config, get_config


# ─── Constants ─────────────────────────────────────────────

OUTPUT_KEYS = ["headgear", "clothing", "shoes", "accessories", "memo"]

# 샘플 테스트 데이터 (실제 데이터가 없을 때 사용)
SAMPLE_TESTS = [
    {
        "name": "hasegawa kobato",
        "outfit_list": [
            {
                "outfit_prompt": "twin tails, gothic lolita dress, black lace headband, white knee-high socks, black mary jane shoes, black ribbon choker",
                "positive_prompt": "1girl, hasegawa kobato, twin tails, gothic lolita dress"
            },
            {
                "outfit_prompt": "twin tails, pink frilly dress, white apron, white hair ribbon, white stockings, pink shoes",
                "positive_prompt": "1girl, hasegawa kobato, twin tails, pink dress"
            },
        ],
        "chat_list": [
            "코바토: 오늘은 밖에 나가야 해... 이 옷을 입어야지. 하아... 귀찮아...\n<!--[asmd]--><img src='test.png'><!--[/asmd]-->"
        ]
    },
    {
        "name": "kashiwazaki sena",
        "outfit_list": [
            {
                "outfit_prompt": "long blonde hair, blue butterfly hair ornament, school uniform, white blouse, blue pleated skirt, black knee-high socks, brown loafers",
                "positive_prompt": "1girl, kashiwazaki sena, blonde hair, butterfly ornament, school uniform"
            },
        ],
        "chat_list": [
            "세나: 나는 역시 이 옷이 잘 어울리지 않아? 후후, 당연하지만!\n<lb-xnai mode='inlay'>test</lb-xnai>"
        ]
    },
    {
        "name": "kusunoki yukimura",
        "outfit_list": [
            {
                "outfit_prompt": "short black hair, maid dress, white apron, white headband, black stockings, black shoes",
                "positive_prompt": "1girl, kusunoki yukimura, maid dress"
            },
            {
                "outfit_prompt": "short black hair, casual clothes, white t-shirt, black shorts, sneakers",
                "positive_prompt": "1girl, kusunoki yukimura, casual"
            },
        ],
        "chat_list": [
            "유키무라: 안녕하세요, 선배. 오늘은 이 옷으로 왔습니다."
        ]
    },
    {
        "name": "unknown",
        "outfit_list": [
            {
                "outfit_prompt": "casual clothes, t-shirt, jeans",
                "positive_prompt": "1girl, casual"
            },
        ],
        "chat_list": []
    },
]


# ─── Data Loading ──────────────────────────────────────────

def load_real_data():
    """workflow_backup/mode/outfit_mode/*/data.json 에서 테스트 데이터 로드"""
    data_dir = os.path.join(PROJECT_ROOT, "workflow_backup", "mode", "outfit_mode")
    if not os.path.isdir(data_dir):
        return []

    test_cases = []
    for char_dir_name in os.listdir(data_dir):
        char_dir = os.path.join(data_dir, char_dir_name)
        data_path = os.path.join(char_dir, "data.json")

        if not os.path.isfile(data_path):
            continue

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            outfit_list = []
            chat_list = []
            for entry in data.get("entries", []):
                outfit_list.append({
                    "outfit_prompt": entry.get("outfit_prompt", ""),
                    "positive_prompt": entry.get("positive_prompt", ""),
                })
                chat_content = entry.get("chat_content", "")
                if chat_content:
                    chat_list.append(chat_content)

            test_cases.append({
                "name": data["name"],
                "outfit_list": outfit_list,
                "chat_list": chat_list,
                "llm_result": data.get("llm_result"),  # 이전 LLM 결과
            })
        except Exception as e:
            print(f"[WARN] {data_path} 로드 실패: {e}")

    return test_cases


# ─── Output Validation ────────────────────────────────────

def validate_output(result_text: str) -> dict:
    """LLM 출력 검증. 검증 리포트 반환."""
    report = {
        "raw_length": len(result_text),
        "is_json": False,
        "has_all_keys": False,
        "missing_keys": [],
        "extra_keys": [],
        "parsed": None,
        "error": None,
        "note": None,
    }

    # LLM 실패 메시지 확인
    if result_text.startswith("[LLM 실패]"):
        report["error"] = result_text
        report["note"] = "LLM call failed"
        return report

    # JSON 파싱 시도
    parsed = None

    # 1차: 직접 파싱
    try:
        parsed = json.loads(result_text)
        report["is_json"] = True
    except json.JSONDecodeError:
        pass

    # 2차: 마크다운 코드 블록에서 추출
    if parsed is None:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                report["is_json"] = True
                report["note"] = "Extracted from markdown code block"
            except json.JSONDecodeError:
                pass

    # 3차: 원시 JSON 추출
    if parsed is None:
        match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                report["is_json"] = True
                report["note"] = "Extracted raw JSON from text"
            except json.JSONDecodeError:
                pass

    if not report["is_json"]:
        report["error"] = "Output is not valid JSON"
        return report

    # 키 검증
    found_keys = set(parsed.keys())
    required = set(OUTPUT_KEYS)
    report["missing_keys"] = list(required - found_keys)
    report["extra_keys"] = list(found_keys - required)
    report["has_all_keys"] = len(report["missing_keys"]) == 0
    report["parsed"] = parsed

    return report


# ─── Prompt Module Loader ─────────────────────────────────

def load_prompt_module(prompt_file: str):
    """customprompt/ 에서 프롬프트 모듈 동적 로드"""
    prompt_path = os.path.join(PROJECT_ROOT, "customprompt", prompt_file)
    if not os.path.isfile(prompt_path):
        raise FileNotFoundError(f"프롬프트 파일 없음: {prompt_path}")

    spec = importlib.util.spec_from_file_location("prompt_module", prompt_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'run'):
        raise AttributeError("프롬프트 모듈에 async 'run' 함수가 없습니다")

    return module


# ─── Color Output ──────────────────────────────────────────

class C:
    """터미널 컬러 코드"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def green(text): return f"{C.GREEN}{text}{C.RESET}"
    @staticmethod
    def red(text): return f"{C.RED}{text}{C.RESET}"
    @staticmethod
    def yellow(text): return f"{C.YELLOW}{text}{C.RESET}"
    @staticmethod
    def cyan(text): return f"{C.CYAN}{text}{C.RESET}"
    @staticmethod
    def bold(text): return f"{C.BOLD}{text}{C.RESET}"
    @staticmethod
    def dim(text): return f"{C.DIM}{text}{C.RESET}"


# ─── Test Execution ───────────────────────────────────────

async def run_single_test(module, test_case: dict, show_prompt: bool = False) -> dict:
    """단일 테스트 케이스 실행"""
    name = test_case["name"]
    outfit_list = test_case["outfit_list"]
    chat_list = test_case.get("chat_list", [])
    previous_result = test_case.get("llm_result")  # 이전 LLM 결과

    print(f"\n{'='*60}")
    print(C.bold(f"  TEST: {name}"))
    print(f"  Outfits: {len(outfit_list)}, Chats: {len(chat_list)}, PreviousResult: {'있음' if previous_result else 'None'}")
    print(f"{'='*60}")

    if show_prompt:
        print(f"\n{C.cyan('[Input Data]')}")
        print(f"  Character: {name}")
        for i, outfit in enumerate(outfit_list, 1):
            op = outfit.get("outfit_prompt", "")
            print(f"  Outfit[{i}]: {op[:120]}{'...' if len(op) > 120 else ''}")
        for i, chat in enumerate(chat_list, 1):
            print(f"  Chat[{i}]: {chat[:120]}{'...' if len(chat) > 120 else ''}")
        if previous_result:
            print(f"  PreviousResult: {previous_result[:120]}{'...' if len(previous_result) > 120 else ''}")

    start_time = time.time()
    try:
        result = await module.run(name, outfit_list, chat_list,
                                  previous_result=previous_result)
        elapsed = time.time() - start_time

        # 특수 케이스: unknown 캐릭터는 스킵
        if result == "LLM 작업 건너뜀":
            print(f"\n{C.yellow('[SKIPPED]')} unknown character - {elapsed:.1f}s")
            return {
                "name": name,
                "success": True,
                "skipped": True,
                "elapsed": elapsed,
                "raw_result": result,
            }

        # LLM 실패 메시지
        if result.startswith("[LLM 실패]"):
            print(f"\n{C.red(f'[LLM FAILED]')} {result[:200]}")
            return {
                "name": name,
                "success": False,
                "skipped": False,
                "elapsed": elapsed,
                "error": result,
                "raw_result": result,
            }

        print(f"\n{C.cyan('[Raw Result]')} ({elapsed:.1f}s)")
        print(C.dim(result[:500] + ("..." if len(result) > 500 else "")))

        # 검증
        validation = validate_output(result)
        print(f"\n{C.cyan('[Validation]')}")
        print(f"  Is JSON:       {C.green('YES') if validation['is_json'] else C.red('NO')}")
        print(f"  Has all keys:  {C.green('YES') if validation['has_all_keys'] else C.red('NO')}")
        if validation['missing_keys']:
            print(f"  Missing keys:  {C.red(str(validation['missing_keys']))}")
        if validation['extra_keys']:
            print(f"  Extra keys:    {C.yellow(str(validation['extra_keys']))}")
        if validation.get('note'):
            print(f"  Note:          {C.yellow(validation['note'])}")
        if validation['error']:
            print(f"  Error:         {C.red(validation['error'])}")

        if validation['parsed']:
            print(f"\n{C.cyan('[Parsed Output]')}")
            for k, v in validation['parsed'].items():
                print(f"  {C.bold(k)}: {v}")

        success = validation['is_json'] and validation['has_all_keys']
        print(f"\n  Result: {C.green('PASS') if success else C.red('FAIL')}")

        return {
            "name": name,
            "success": success,
            "skipped": False,
            "elapsed": elapsed,
            "validation": validation,
            "raw_result": result,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{C.red(f'[ERROR] {e}')}")
        import traceback
        traceback.print_exc()
        return {
            "name": name,
            "success": False,
            "skipped": False,
            "elapsed": elapsed,
            "error": str(e),
        }


async def run_raw_test(prompt_file: str, show_prompt: bool = False):
    """LLM 호출 없이 프롬프트 빌드만 테스트 (raw 모드)"""
    module = load_prompt_module(prompt_file)

    # 모듈에서 빌드 함수 사용 가능한지 확인
    has_system = hasattr(module, '_build_system_prompt')
    has_user = hasattr(module, '_build_user_prompt')

    if not has_system or not has_user:
        print(C.yellow("이 프롬프트 파일은 _build_system_prompt / _build_user_prompt 함수가 없습니다."))
        print("raw 모드를 사용하려면 프롬프트 파일에 해당 함수들이 있어야 합니다.")
        return

    print(f"\n{'='*60}")
    print(C.bold("  RAW PROMPT BUILD TEST"))
    print(f"{'='*60}")

    system_prompt = module._build_system_prompt()
    print(f"\n{C.cyan('[System Prompt]')}")
    print(system_prompt)

    for test_case in SAMPLE_TESTS[:2]:
        name = test_case["name"]
        outfit_list = test_case["outfit_list"]
        chat_list = test_case.get("chat_list", [])

        char_identity = module._get_char_identity(name) if hasattr(module, '_get_char_identity') else "Unknown"

        outfits_parts = []
        for i, entry in enumerate(outfit_list, 1):
            op = entry.get("outfit_prompt", "")
            pp = entry.get("positive_prompt", "")
            if i <= 3 and pp:
                outfits_parts.append(f"Outfit[{i}]: {op}\n  prompt[{i}]: {pp}")
            else:
                outfits_parts.append(f"Outfit[{i}]: {op}")
        outfits_text = "\n".join(outfits_parts)

        chat_context = ""
        if chat_list:
            cleaned = module.clean_chat(chat_list[0]) if hasattr(module, 'clean_chat') else chat_list[0]
            if cleaned:
                chat_context = f"Chat[1]: {cleaned}"

        user_prompt = module._build_user_prompt(name, char_identity, outfits_text, chat_context)

        print(f"\n{'─'*60}")
        print(f"{C.cyan(f'[User Prompt for {name}]')}")
        print(user_prompt)


async def run_all_tests(prompt_file: str = "haganai_extract_outfit_prompt.py",
                        use_real_data: bool = False,
                        show_prompt: bool = False,
                        filter_name: str = None,
                        model_override: str = None):
    """전체 테스트 실행 및 요약"""

    # LLM 설정
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        update_config({
            "llm_service": config.get("llm_service", "copilot"),
            "llm_model": config.get("llm_model", "gpt-4.1"),
            "llm_service2": config.get("llm_service2", ""),
            "llm_model2": config.get("llm_model2", ""),
            "custom_api_url": config.get("custom_api_url", ""),
            "custom_api_url2": config.get("custom_api_url2", ""),
        })

    if model_override:
        update_config({"llm_model": model_override})

    current_config = get_config()
    llm1_info = f"{current_config['llm_service']} / {current_config['llm_model']}"
    llm2_info = ""
    if current_config.get("llm_model2"):
        svc2 = current_config.get("llm_service2") or current_config["llm_service"]
        llm2_info = f" | LLM2: {svc2} / {current_config['llm_model2']}"
    print(C.bold(f"\nLLM1: {llm1_info}{llm2_info}"))

    # 프롬프트 모듈 로드
    print(f"Loading prompt: {prompt_file}")
    module = load_prompt_module(prompt_file)

    # 테스트 데이터 로드
    if use_real_data:
        test_cases = load_real_data()
        if not test_cases:
            print(C.yellow("실제 데이터 없음. 샘플 데이터 사용."))
            test_cases = SAMPLE_TESTS
        else:
            print(f"실제 데이터 {len(test_cases)}개 로드 완료.")
    else:
        test_cases = SAMPLE_TESTS

    # 이름 필터
    if filter_name:
        test_cases = [tc for tc in test_cases if filter_name.lower() in tc["name"].lower()]
        if not test_cases:
            print(C.red(f"'{filter_name}' 매칭되는 테스트 케이스 없음"))
            return

    # 테스트 실행
    results = []
    for tc in test_cases:
        result = await run_single_test(module, tc, show_prompt)
        results.append(result)

    # 요약
    print(f"\n{'='*60}")
    print(C.bold(f"  SUMMARY ({len(results)} tests)"))
    print(f"{'='*60}")

    passed = 0
    failed = 0
    skipped = 0

    for r in results:
        if r.get("skipped"):
            skipped += 1
            print(f"  {C.yellow('[SKIP]')} {r['name']}")
        elif r["success"]:
            passed += 1
            print(f"  {C.green('[PASS]')} {r['name']} ({r['elapsed']:.1f}s)")
        else:
            failed += 1
            err = r.get("error", "format error")
            if err and len(err) > 80:
                err = err[:80] + "..."
            print(f"  {C.red('[FAIL]')} {r['name']} ({r['elapsed']:.1f}s) - {err}")

    print(f"\n  Total: {len(results)}  |  Passed: {C.green(str(passed))}  |  "
          f"Failed: {C.red(str(failed))}  |  Skipped: {C.yellow(str(skipped))}")

    # 결과 JSON 저장
    if results:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"test_{int(time.time())}.json")

        save_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_file": prompt_file,
            "model": current_config["llm_model"],
            "model2": current_config.get("llm_model2", ""),
            "results": []
        }
        for r in results:
            entry = {
                "name": r["name"],
                "success": r["success"],
                "elapsed": round(r["elapsed"], 2),
            }
            if r.get("validation"):
                entry["is_json"] = r["validation"]["is_json"]
                entry["has_all_keys"] = r["validation"]["has_all_keys"]
                if r["validation"]["parsed"]:
                    entry["parsed_output"] = r["validation"]["parsed"]
            if r.get("error"):
                entry["error"] = r["error"]
            if r.get("raw_result"):
                entry["raw_result"] = r["raw_result"]
            save_data["results"].append(entry)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved: {save_path}")


# ─── CLI ───────────────────────────────────────────────────

def print_usage():
    print(f"""
{C.bold('Outfit Prompt Test Runner')}
{'='*40}

Usage:
  python main.py [options]

Options:
  --prompt FILE       Prompt file to test (default: haganai_extract_outfit_prompt.py)
  --real              Use real data from workflow_backup/mode/outfit_mode/
  --show              Show input data details
  --filter NAME       Filter test cases by character name
  --model MODEL       Override LLM model (e.g. gpt-4.1, gemini-3-flash-preview)
  --raw               Prompt build only (no LLM call)
  --help              Show this help

Examples:
  python main.py                          # Run with sample data
  python main.py --real                   # Run with real data
  python main.py --filter kobato          # Test only kobato
  python main.py --prompt my_prompt.py    # Test specific prompt file
  python main.py --raw                    # Show prompt without calling LLM
""")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_usage()
        sys.exit(0)

    prompt_file = "haganai_extract_outfit_prompt.py"
    use_real_data = False
    show_prompt = False
    filter_name = None
    model_override = None
    raw_mode = False

    i = 0
    while i < len(args):
        if args[i] == "--prompt" and i + 1 < len(args):
            prompt_file = args[i + 1]
            i += 2
        elif args[i] == "--real":
            use_real_data = True
            i += 1
        elif args[i] == "--show":
            show_prompt = True
            i += 1
        elif args[i] == "--filter" and i + 1 < len(args):
            filter_name = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif args[i] == "--raw":
            raw_mode = True
            i += 1
        else:
            print(f"Unknown option: {args[i]}")
            print_usage()
            sys.exit(1)

    if raw_mode:
        asyncio.run(run_raw_test(prompt_file, show_prompt))
    else:
        asyncio.run(run_all_tests(prompt_file, use_real_data, show_prompt, filter_name, model_override))
