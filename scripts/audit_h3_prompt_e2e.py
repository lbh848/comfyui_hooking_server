"""Live, text-only E2E audit for the REF2V planner and final H3 writer.

This script reads the deployed LLM configuration, injects it into this process only,
and calls the same message builders and task routes used by VideoMode. It never writes
config.json, key files, production histories, or video assets. Synthetic natural-language
reference descriptions stand in for image payloads so the audit remains LLM-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from modes import llm_service
from modes.video_mode import (
    VideoMode,
    compose_h3_prompt_candidate,
    h3_candidate_selection_messages,
    h3_independent_candidate_messages,
    normalize_instruction_draft,
    parse_h3_candidate_selection,
    validate_h3_prompt_candidate,
    validate_instruction_draft,
    validate_ref2v_prompt_body,
)


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    duration: int
    picture_count: int
    direction: str
    visual_context: str
    expected_semantics: str
    allow_camera_motion: bool = True
    allow_background_change: bool = False


CASES = (
    AuditCase(
        case_id="combat_target_spear",
        duration=12,
        picture_count=2,
        direction=(
            "Create a cinematic fight between the two referenced women. Begin with only "
            "Picture 1's swordswoman on screen, holding her own short sword in her right "
            "hand in an outward guard toward the off-screen opponent. Picture 2's woman "
            "then enters using one newly introduced long spear with one-handed spear "
            "technique, continuously controlled by her right hand while her left hand "
            "remains free. Show a "
            "readable continuous exchange, then finish with Picture 1's short sword blade "
            "pressing against the front portion of Picture 2's much longer spear shaft just "
            "below its spearhead. Scene changes and close views are allowed but not required."
        ),
        visual_context=(
            "<Picture 1> depicts a female swordswoman holding one short gold-hilted sword "
            "in her right hand against an incidental white-to-yellow character-card "
            "backdrop. <Picture 2> depicts a second woman with both hands empty, pressing "
            "her cheeks with her fingers against an incidental solid-white character-card "
            "backdrop; no spear or sword is visible in Picture 2. The two women have distinct "
            "faces, clothing, and silhouettes."
        ),
        expected_semantics=(
            "The spear is target-video content rather than a Picture 2 trait. The first woman "
            "alone keeps one short sword and the second woman alone keeps one long spear; no "
            "duplication, dual wielding, or transfer occurs. The requested one-handed spear "
            "technique remains controlled by the second woman's right hand. The opening sword points away "
            "from its holder toward the unseen opponent. The exchange has multiple connected "
            "action phases, and the final broad shaft-to-blade lock keeps both weapons and "
            "controllers visibly distinct without hidden close-up setup changes. The final lock "
            "settles only long enough to read and does not replace a large fraction of the duel."
        ),
    ),
    AuditCase(
        case_id="single_person_two_handed_box",
        duration=8,
        picture_count=2,
        direction=(
            "Picture 1's warehouse worker lifts the long sealed box from Picture 2 with both "
            "hands, carries it across the loading area, adjusts his grip once while keeping "
            "the box supported, and sets the same box securely on a waist-high cart."
        ),
        visual_context=(
            "<Picture 1> depicts one empty-handed warehouse worker. <Picture 2> depicts one "
            "long sealed cardboard box with a blue shipping label, isolated as an object "
            "reference. Neither picture defines the target location."
        ),
        expected_semantics=(
            "One continuous box instance is controlled by both hands of the same worker. The "
            "grip adjustment preserves support, and the final set-down visibly transfers "
            "support to the cart before the hands release. Two-handed control must not be "
            "simplified to one hand or misread as duplication."
        ),
    ),
    AuditCase(
        case_id="joint_glass_carry",
        duration=10,
        picture_count=3,
        direction=(
            "Picture 1's installer and Picture 2's installer jointly carry the single large "
            "glass pane from Picture 3 through a doorway. Each keeps both hands on their own "
            "opposite edge. They pause to tilt the pane together, clear the frame, level it, "
            "and continue without either person taking sole control."
        ),
        visual_context=(
            "<Picture 1> and <Picture 2> depict two visually distinct empty-handed installers. "
            "<Picture 3> depicts one rectangular transparent glass pane with a green-tinted "
            "edge, isolated as an object reference."
        ),
        expected_semantics=(
            "The one pane has a two-person controller set throughout. Both participants' two "
            "hand contacts remain separately readable during the coordinated tilt and leveling. "
            "No false handoff, sole ownership, extra pane, or lost support is introduced."
        ),
    ),
    AuditCase(
        case_id="explicit_cup_handoff",
        duration=8,
        picture_count=3,
        direction=(
            "Picture 1's barista hands the single ceramic cup from Picture 3 across the counter "
            "to Picture 2's customer. Show the customer establish a secure grip before the "
            "barista releases, then the customer brings that same cup closer and takes one sip."
        ),
        visual_context=(
            "<Picture 1> depicts an empty-handed barista. <Picture 2> depicts an empty-handed "
            "customer. <Picture 3> depicts one distinctive white ceramic cup with a red rim, "
            "isolated as an object reference."
        ),
        expected_semantics=(
            "Exactly one cup transfers through an observable overlap of support, release, and "
            "new secure control. The barista must not keep controlling it after release, and "
            "the customer must not sip before the transfer is physically complete."
        ),
    ),
    AuditCase(
        case_id="independent_nearby_props",
        duration=9,
        picture_count=2,
        direction=(
            "Picture 1's street performer crosses Picture 2's courier in a busy plaza. The "
            "performer continuously carries one short red juggling club in her left hand; the "
            "courier continuously carries one long blue rolled poster tube under his right arm. "
            "They sidestep around each other without touching, exchanging, or using the props."
        ),
        visual_context=(
            "<Picture 1> depicts the female performer and the short red club already held in "
            "her left hand. <Picture 2> depicts the male courier and the long blue poster tube "
            "already tucked under his right arm. Their incidental portrait backdrops are not "
            "locations."
        ),
        expected_semantics=(
            "The nearby objects keep distinct shape, color, scale, count, controller, and body "
            "attachment. Passing close must not merge them, swap them, add shared contact, or "
            "turn either into a different prop."
        ),
    ),
    AuditCase(
        case_id="target_only_long_paint_roller",
        duration=9,
        picture_count=1,
        direction=(
            "Picture 1's painter enters the target scene carrying one newly introduced long "
            "paint roller with both hands. She keeps its roller head outward toward the wall, "
            "rolls one tall stripe from low to high, lowers it along a clear path away from her "
            "body, and finishes with the tool still under secure two-handed control."
        ),
        visual_context=(
            "<Picture 1> depicts an empty-handed female painter in recognizable work clothes "
            "against a blank catalog backdrop. No roller, pole, wall, or target environment is "
            "shown."
        ),
        expected_semantics=(
            "The roller is target-only and must not be claimed as preserved from Picture 1. One "
            "stable long tool keeps two-handed control, an outward working end, a continuous "
            "wall-directed path, and body clearance without multiplying or reversing."
        ),
    ),
    AuditCase(
        case_id="incidental_background_not_location",
        duration=8,
        picture_count=2,
        direction=(
            "Stage Picture 1's dancer wearing the costume design from Picture 2 on a rain-wet "
            "city rooftop at night. She performs one compact turn, catches her balance, and "
            "finishes looking across the skyline while rain and reflected signs remain coherent."
        ),
        visual_context=(
            "<Picture 1> depicts the dancer against a plain white studio sweep. <Picture 2> is "
            "a costume design card against a beige gradient. These are incidental presentation "
            "backdrops, not environment references."
        ),
        expected_semantics=(
            "The target location is the requested rainy rooftop, not either blank or gradient "
            "reference backdrop. The dancer identity and assigned costume design are preserved, "
            "the single turn supports the action, and the rooftop remains spatially continuous."
        ),
    ),
    AuditCase(
        case_id="assigned_environment_reference",
        duration=10,
        picture_count=2,
        direction=(
            "Use Picture 1 only for the chef's identity and clothing, and Picture 2 as the "
            "authoritative dining-car environment. In that dining car, the chef prepares one "
            "tableside dessert through a clear sequence of arranging, pouring, folding, and "
            "presenting it while the train continues moving."
        ),
        visual_context=(
            "<Picture 1> depicts a chef against a gray studio card. <Picture 2> depicts an ornate "
            "vintage train dining car with green booths, brass lamps, a narrow aisle, and night "
            "visible through its windows; no person is present there."
        ),
        expected_semantics=(
            "Picture 2 is intentionally defined and retained as the target environment while "
            "Picture 1's gray backdrop is ignored. The chef performs the requested causal food "
            "preparation rather than unrelated spectacle, and the dining-car layout stays stable."
        ),
    ),
    AuditCase(
        case_id="sustained_pottery_progression",
        duration=12,
        picture_count=1,
        direction=(
            "Picture 1's potter shapes one clay vessel for the full video: first centers the "
            "spinning clay, opens the middle, raises the walls, then refines the rim and settles "
            "with the finished wet vessel still turning. Prioritize the continuous craft process."
        ),
        visual_context=(
            "<Picture 1> depicts the potter's identity and apron in a neutral portrait. No wheel, "
            "clay, vessel, workshop, or action pose is shown."
        ),
        expected_semantics=(
            "One target-only clay mass develops causally into one vessel through all requested "
            "phases. The prompt spends the duration on hand pressure, support, material response, "
            "and visible shape change instead of entrances, reaction inserts, many camera cuts, "
            "or a long static payoff."
        ),
    ),
    AuditCase(
        case_id="camera_permission_not_quota",
        duration=8,
        picture_count=2,
        direction=(
            "Picture 1's friend quietly pours tea for Picture 2's friend, then they lift their "
            "cups in a small toast. Camera movement and close views are allowed but not required; "
            "keep the intimate action coherent and easy to follow."
        ),
        visual_context=(
            "<Picture 1> and <Picture 2> depict two distinct friends in separate portrait cards. "
            "Neither card assigns a target location or target props."
        ),
        expected_semantics=(
            "Camera permission must not become a requirement to use many viewpoints. Pouring, cup "
            "control, and the toast remain the primary progression, and each actual viewpoint is "
            "an explicit shot rather than multiple hidden setups inside one shot."
        ),
    ),
    AuditCase(
        case_id="explicit_three_setups",
        duration=9,
        picture_count=1,
        direction=(
            "Use exactly three camera setups for Picture 1's watchmaker repairing a pocket watch: "
            "an opening wide view of the bench, an overhead view as the case is opened and the "
            "mechanism adjusted, and a final close view as the same watch begins ticking."
        ),
        visual_context=(
            "<Picture 1> depicts the watchmaker's identity, clothing, and one distinctive engraved "
            "pocket watch held at a workbench."
        ),
        expected_semantics=(
            "The final prompt has exactly three numbered shots corresponding to the three requested "
            "setups, with no hidden angle or scale changes inside them. The same single engraved "
            "watch remains controlled and develops from closed to adjusted to ticking."
        ),
    ),
    AuditCase(
        case_id="intentional_self_contact",
        duration=7,
        picture_count=1,
        direction=(
            "Picture 1's performer uses one newly introduced makeup brush in her right hand to "
            "apply a deliberate stripe to her own left cheek, pulls the brush away, checks the "
            "result in a target-scene mirror, and smiles."
        ),
        visual_context=(
            "<Picture 1> depicts the performer's identity and costume with both hands empty. No "
            "brush, mirror, or target environment is visible."
        ),
        expected_semantics=(
            "The explicitly requested self-directed brush contact is preserved rather than blocked "
            "by generic body-clearance guidance. The brush remains one target-only object controlled "
            "by the right hand, contacts only the intended left cheek, then visibly separates."
        ),
    ),
    AuditCase(
        case_id="quiet_emotional_reunion",
        duration=12,
        picture_count=2,
        direction=(
            "Picture 1's elderly father waits alone on a dim apartment landing. Picture 2's "
            "adult daughter comes up the stairs, they recognize each other after a hesitant "
            "beat, and the distance closes into one restrained embrace. Keep it intimate and "
            "emotionally specific, with no melodramatic spectacle."
        ),
        visual_context=(
            "<Picture 1> depicts the father's identity and modest brown coat in a portrait. "
            "<Picture 2> depicts the daughter's identity, navy travel jacket, and worn canvas "
            "bag in a separate portrait. Neither portrait supplies the apartment location."
        ),
        expected_semantics=(
            "The scene earns its embrace through waiting, mutual recognition, hesitation, and "
            "a visible decision to close the distance. Intimacy comes from performance, spatial "
            "staging, and restraint rather than excess cutting, generic smiling, or invented drama."
        ),
    ),
    AuditCase(
        case_id="suspense_reveal_single_take",
        duration=10,
        picture_count=1,
        direction=(
            "In one uninterrupted moving shot, follow Picture 1's night guard down a museum "
            "corridor as a faint scraping sound makes her slow down. She rounds the final display "
            "case and discovers that one target-scene pedestal is empty. End on her contained "
            "realization, not a monster or chase."
        ),
        visual_context=(
            "<Picture 1> depicts the night guard's identity, dark green uniform, flashlight, and "
            "key ring against an incidental studio backdrop. No museum or missing exhibit is shown."
        ),
        expected_semantics=(
            "One continuous setup builds suspense through sound, pace, gaze, occlusion, and delayed "
            "visual information. The empty pedestal is the payoff and her restrained realization "
            "settles it; cuts, a creature, a chase, or an unrelated scare would weaken the request."
        ),
    ),
    AuditCase(
        case_id="comedic_cooking_timing",
        duration=9,
        picture_count=2,
        direction=(
            "Picture 1's proud home cook flips one pancake too high. Picture 2's sleepy roommate "
            "walks into the kitchen exactly as it lands neatly on the plate they are carrying. "
            "Let both register the impossible success, then end on the cook pretending it was "
            "intentional. Play the comedy through timing and reactions, not frantic montage."
        ),
        visual_context=(
            "<Picture 1> depicts the cook's identity and striped apron. <Picture 2> depicts the "
            "roommate's identity, pajamas, and one plain empty plate. Their portrait backdrops do "
            "not define the target kitchen."
        ),
        expected_semantics=(
            "The single pancake follows a readable launch and landing path synchronized with the "
            "roommate's entrance and existing plate. The shared realization and the cook's false "
            "confidence form the comic payoff; extra mishaps or rapid reaction inserts do not."
        ),
    ),
    AuditCase(
        case_id="dance_phrase_momentum",
        duration=12,
        picture_count=2,
        direction=(
            "Picture 1's lead dancer and Picture 2's partner perform one flowing contemporary duet "
            "phrase across a bare rehearsal floor: a mirrored reach becomes a passing turn, their "
            "forearms connect to redirect momentum, and they separate into opposite low finishes. "
            "Preserve continuous weight transfer and make the phrase feel musical rather than posed."
        ),
        visual_context=(
            "<Picture 1> and <Picture 2> depict two distinct dancers and their fitted rehearsal "
            "clothes in separate neutral portraits. Neither image establishes a target pose or floor."
        ),
        expected_semantics=(
            "The duet is one connected movement phrase whose reach, turn, forearm contact, redirection, "
            "and separation carry momentum forward. It should not read as isolated poses, simultaneous "
            "whole-body starts, arbitrary flourishes, or edits substituting for choreography."
        ),
    ),
    AuditCase(
        case_id="product_reveal_restraint",
        duration=8,
        picture_count=1,
        direction=(
            "Create a premium product reveal for the single wristwatch in Picture 1. Begin with its "
            "brushed case barely catching a narrow moving reflection, let the light reveal the dial "
            "and second hand in one deliberate progression, and finish on a clean three-quarter hero "
            "view. Keep the watch stationary and avoid gratuitous cuts or invented features."
        ),
        visual_context=(
            "<Picture 1> is an isolated object reference for one steel wristwatch with a charcoal "
            "dial, fine white indices, a date window, and a brushed metal bracelet on a plain backdrop."
        ),
        expected_semantics=(
            "The same stationary watch is revealed by motivated light progression from partial case "
            "information to readable dial detail to a complete hero view. Premium restraint, material "
            "specificity, and a clear visual payoff matter more than camera activity or added features."
        ),
    ),
    AuditCase(
        case_id="atmospheric_observation_no_forced_climax",
        duration=10,
        picture_count=1,
        direction=(
            "Picture 1's child quietly watches the first snow through a late-afternoon classroom "
            "window. Nothing dramatic happens: breath softly fogs a small patch of glass, one finger "
            "traces a short line through it, and the child settles into a private smile as flakes "
            "continue outside. Hold the contemplative tone."
        ),
        visual_context=(
            "<Picture 1> depicts the child's identity, school cardigan, and short dark hair in a "
            "neutral portrait. No classroom, snow, window, or action pose is supplied by the image."
        ),
        expected_semantics=(
            "The requested small-scale progression remains quiet and observational. Fog, traced line, "
            "continued snowfall, attention, and the private smile provide sufficient development; the "
            "prompt must not manufacture a dramatic climax merely to appear cinematic."
        ),
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        print(f"[H3_E2E] JSON 읽기 실패: path={path}", flush=True)
        traceback.print_exc()
        raise
    if not isinstance(value, dict):
        print(f"[H3_E2E] JSON 루트 객체 오류: path={path}, type={type(value).__name__}", flush=True)
        raise ValueError(f"JSON object required: {path}")
    return value


async def _run_case(case: AuditCase) -> dict[str, Any]:
    print(f"[H3_E2E] START {case.case_id}", flush=True)
    started = time.time()
    planner = ""
    candidates: list[str] = []
    writer_raw_debug: list[str] = []
    try:
        planner_started = time.time()
        planner_messages = VideoMode._instruction_direct_messages(
            "ref2v",
            "en",
            duration=case.duration,
            user_input=case.direction,
            allow_camera_motion=case.allow_camera_motion,
            allow_background_change=case.allow_background_change,
        )
        planner_messages[-1]["content"] += (
            "\n\nTEST-ONLY TEXTUAL SUBSTITUTE FOR THE REFERENCE IMAGES SUPPLIED TO "
            "THE PRODUCTION VISION CALL — treat these as visible image facts, not as "
            "additional creative direction:\n"
            + case.visual_context
        )
        planner_execution = await llm_service.callLLMTaskResult(
            "video_prompt_ref2v",
            planner_messages,
            result_validator=validate_instruction_draft,
        )
        planner = normalize_instruction_draft(planner_execution.text)
        if not planner_execution.accepted or not planner:
            raise RuntimeError(
                "planner returned no accepted direction: "
                f"reason={planner_execution.reason}, raw={planner_execution.raw_response!r}"
            )
        planner_elapsed = time.time() - planner_started

        writer_messages = VideoMode._prompt_messages(
            "ref2v",
            planner,
            visual_context="visual_context:\n" + case.visual_context,
            duration=case.duration,
            picture_count=case.picture_count,
        )
        candidate_started = time.time()
        compose_slot = llm_service.routing_primary_slot(
            "video_prompt_ref2v_compose"
        )

        async def call_nonempty_text(
            messages: list[dict],
            *,
            execution_id: str,
            response_label: str,
        ):
            """Retry once only when the call produced no response text."""

            last_execution = None
            for attempt in range(1, 3):
                last_execution = await llm_service.callLLMTaskResult(
                    "video_prompt_ref2v_compose",
                    messages,
                    result_validator=lambda value: validate_h3_prompt_candidate(
                        value,
                        "ref2v",
                    ),
                    execution_id=execution_id,
                    force_slot=compose_slot,
                )
                if last_execution.accepted:
                    return last_execution
                if attempt == 1:
                    print(
                        "[H3_E2E] no response text; retrying once: "
                        f"case={case.case_id}, response={response_label}, "
                        f"reason={last_execution.reason}",
                        flush=True,
                    )
            assert last_execution is not None
            print(
                "[H3_E2E] no response text after two calls: "
                f"case={case.case_id}, response={response_label}, "
                f"reason={last_execution.reason}",
                flush=True,
            )
            raise RuntimeError(
                f"{response_label} returned no response text after two calls"
            )

        async def generate_candidate(candidate_number: int) -> dict[str, Any]:
            candidate_messages = h3_independent_candidate_messages(
                writer_messages,
                candidate_number,
            )
            execution = await call_nonempty_text(
                candidate_messages,
                execution_id=f"h3_e2e:{case.case_id}:candidate_{candidate_number}",
                response_label=f"candidate_{candidate_number}",
            )
            raw = str(execution.raw_response or "").strip()
            prompt = compose_h3_prompt_candidate(
                execution.text,
                "ref2v",
                case.duration,
            )
            return {"number": candidate_number, "prompt": prompt, "raw": raw}

        candidate_results = await asyncio.gather(
            *(generate_candidate(index) for index in range(1, 4))
        )
        candidate_results = sorted(
            candidate_results,
            key=lambda item: int(item["number"]),
        )
        candidates = [str(item["prompt"]) for item in candidate_results]
        writer_raw_debug = [str(item["raw"]) for item in candidate_results]
        candidate_elapsed = time.time() - candidate_started

        selector_messages = h3_candidate_selection_messages(
            mode="ref2v",
            instruction=planner,
            visual_context="visual_context:\n" + case.visual_context,
            candidates=candidates,
        )
        selector_started = time.time()
        selector_execution = await call_nonempty_text(
            selector_messages,
            execution_id=f"h3_e2e:{case.case_id}:selector",
            response_label="selector",
        )
        selection_text = str(selector_execution.text or "").strip()
        try:
            selected_candidate = parse_h3_candidate_selection(
                selection_text,
                len(candidates),
            )
        except Exception:
            print(
                "[H3_E2E] selector number missing; using candidate 1 without recall: "
                f"case={case.case_id}, response={selection_text[:500]!r}",
                flush=True,
            )
            traceback.print_exc()
            selected_candidate = 1
        selector_elapsed = time.time() - selector_started
        final_prompt = candidates[selected_candidate - 1]
        format_pass, format_reason = validate_ref2v_prompt_body(
            final_prompt,
            case.picture_count,
            case.duration,
        )
        elapsed = time.time() - started
        print(
            f"[H3_E2E] SELECTED {case.case_id} candidate={selected_candidate} "
            f"format={format_pass} elapsed={elapsed:.2f}s",
            flush=True,
        )
        return {
            "case": asdict(case),
            "planner": planner,
            "candidates": candidates,
            "selection": selection_text,
            "selected_candidate": selected_candidate,
            "final_prompt": final_prompt,
            "format": {"pass": format_pass, "reason": format_reason},
            "timing_seconds": {
                "planner": round(planner_elapsed, 3),
                "parallel_candidates": round(candidate_elapsed, 3),
                "selector": round(selector_elapsed, 3),
                "total": round(elapsed, 3),
            },
            "pipeline_pass": True,
            "manual_review_required": True,
            "error": "",
        }
    except Exception as exc:
        print(f"[H3_E2E] CASE failed: case={case.case_id}, error={exc}", flush=True)
        traceback.print_exc()
        return {
            "case": asdict(case),
            "planner": planner,
            "candidates": candidates,
            "final_prompt": "",
            "writer_raw": writer_raw_debug,
            "format": {"pass": False, "reason": str(exc)},
            "pipeline_pass": False,
            "manual_review_required": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _select_cases(case_ids: list[str]) -> list[AuditCase]:
    if not case_ids:
        return list(CASES)
    available = {case.case_id: case for case in CASES}
    unknown = sorted(set(case_ids) - set(available))
    if unknown:
        print(f"[H3_E2E] 알 수 없는 case: {unknown}", flush=True)
        raise ValueError(f"unknown cases: {unknown}")
    return [available[case_id] for case_id in case_ids]


async def _main_async(args: argparse.Namespace) -> int:
    config = _read_json(PROJECT_ROOT / "config.json")
    keys = _read_json(PROJECT_ROOT / "key" / "llm_keys.json")

    # Keep all LLM configuration and concurrency changes process-local. Prevent the
    # service from appending this audit to production logs or histories.
    llm_service._llm_log = lambda _message: None
    llm_service._log_history = lambda *_args, **_kwargs: None
    llm_service.update_config({**config, **keys})
    if args.concurrency:
        for number in range(1, llm_service.LLM_SLOT_COUNT + 1):
            suffix = "" if number == 1 else str(number)
            key = f"llm_max_concurrency{suffix}"
            if key in llm_service._current_config:
                llm_service._current_config[key] = int(args.concurrency)
        llm_service._wake_request_limit_waiters()

    selected = _select_cases(args.cases)
    print(
        f"[H3_E2E] cases={len(selected)} candidate_count=3 "
        f"process_concurrency={args.concurrency or 'config'}",
        flush=True,
    )
    results = await asyncio.gather(*(_run_case(case) for case in selected))

    passed = sum(1 for result in results if result.get("pipeline_pass") is True)
    failed = len(results) - passed
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "LLM-only REF2V planner -> three parallel natural-language candidates -> numeric selection -> verbatim selected candidate",
        "production_files_modified": False,
        "planner_route": (config.get("llm_routing") or {}).get("video_prompt_ref2v", {}),
        "writer_route": (config.get("llm_routing") or {}).get("video_prompt_ref2v_compose", {}),
        "semantic_verdict_source": "manual reading of the selected final prompt",
        "summary": {"total": len(results), "passed": passed, "failed": failed},
        "results": results,
    }

    output_path = (
        Path(args.output).resolve()
        if args.output
        else PROJECT_ROOT
        / ".work"
        / "h3_prompt_e2e"
        / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )
    if output_path.exists():
        print(f"[H3_E2E] 기존 결과 파일 덮어쓰기 거부: path={output_path}", flush=True)
        return 2
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        print(f"[H3_E2E] 결과 저장 실패: path={output_path}", flush=True)
        traceback.print_exc()
        return 2

    print(
        f"[H3_E2E] COMPLETE pass={passed}/{len(results)} failed={failed} report={output_path}",
        flush=True,
    )
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[],
        help="Run only the listed case ids; omit to run the full matrix.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Process-local per-slot request limit. No config file is modified.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.concurrency < 1:
        print(f"[H3_E2E] concurrency는 1 이상이어야 합니다: {args.concurrency}", flush=True)
        return 2
    try:
        return asyncio.run(_main_async(args))
    except Exception:
        print("[H3_E2E] 실행 중 예외가 발생했습니다", flush=True)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
