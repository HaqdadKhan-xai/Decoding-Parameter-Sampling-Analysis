from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from config import (
    CONFIGURATIONS,
    MODEL_TAGS,
    PLANNED_GENERATION_COUNT,
    DecodingConfig,
)
from prompts import PROMPTS, PROMPTS_BY_ID


OLLAMA_BASE_URL = "http://127.0.0.1:11434"
SCHEMA_VERSION = "revised-otat-v1"
RAW_RESULTS_DIR = Path(__file__).parent / "results" / "raw"
TEST_PROMPT_ID = "D1"
TEST_MODEL = "qwen2.5:3b"
TEST_CONFIG = DecodingConfig("baseline", temperature=0.0, top_k=1, top_p=0.3)
REQUEST_TIMEOUT_SECONDS = 300
MAX_TRANSIENT_RETRIES = 2
EXPERIMENT_RESULTS_PATH = RAW_RESULTS_DIR / "experiment_generations.jsonl"
REVISED_RESULTS_PATH = RAW_RESULTS_DIR / "revised_experiment_generations.jsonl"


def utc_timestamp() -> str:
    """Return an unambiguous UTC timestamp for result metadata."""
    return datetime.now(timezone.utc).isoformat()


def result_path(started_at: datetime) -> Path:
    """Build a unique result filename without relying on an existing index."""
    filename = (
        f"test_{started_at.strftime('%Y%m%dT%H%M%S_%fZ')}_"
        f"{TEST_PROMPT_ID}_{TEST_MODEL.replace(':', '-')}.json"
    )
    return RAW_RESULTS_DIR / filename


def verify_ollama_and_model(session: requests.Session) -> str:
    """Verify the local service and both required exact model tags."""
    version_response = session.get(f"{OLLAMA_BASE_URL}/api/version", timeout=10)
    version_response.raise_for_status()
    version_payload = version_response.json()
    version = version_payload.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("Ollama version response did not contain a version.")
    tags_response = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    tags_response.raise_for_status()
    payload = tags_response.json()
    installed_tags = {model.get("name") for model in payload.get("models", [])}
    missing_tags = sorted(set(MODEL_TAGS) - installed_tags)
    if missing_tags:
        raise RuntimeError(
            "Required Ollama model tags are missing: " + ", ".join(missing_tags)
        )
    return version


def generation_key(
    prompt_id: str, model: str, configuration: DecodingConfig
) -> str:
    """Return the stable identity of one planned generation."""
    return f"{prompt_id}|{model}|{configuration.key}"


def planned_generations() -> list[tuple[str, str, DecodingConfig]]:
    """Return all planned prompt/model/configuration combinations."""
    return [
        (prompt.prompt_id, model, configuration)
        for prompt in PROMPTS
        for model in MODEL_TAGS
        for configuration in CONFIGURATIONS
    ]


def load_successful_keys(path: Path) -> set[str]:
    """Load successful experiment identities for safe resume support."""
    successful_keys: set[str] = set()
    if not path.exists():
        return successful_keys
    with path.open("r", encoding="utf-8") as records:
        for line_number, line in enumerate(records, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if record.get("run_type") == "experiment" and record.get("status") == "success":
                key = record.get("generation_key")
                if isinstance(key, str):
                    successful_keys.add(key)
    return successful_keys


def append_jsonl(path: Path, result: dict[str, Any]) -> None:
    """Append one complete record and flush it so progress survives interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
        output.flush()


def is_transient_error(error: Exception) -> bool:
    """Identify operational failures that are safe to retry."""
    if isinstance(error, requests.Timeout):
        return True
    if isinstance(error, requests.ConnectionError):
        return True
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code in {408, 429, 500, 502, 503, 504}
    return False


def execute_generation(
    session: requests.Session,
    prompt_id: str,
    model: str,
    configuration: DecodingConfig,
    prompt_text: str | None = None,
    num_predict: int | None = None,
) -> dict[str, Any]:
    """Execute one generation, with bounded retries for transient failures."""
    prompt = PROMPTS_BY_ID.get(prompt_id)
    selected_prompt = prompt_text if prompt_text is not None else (
        prompt.text if prompt is not None else None
    )
    if selected_prompt is None:
        raise ValueError(f"Unknown prompt ID: {prompt_id}")
    key = generation_key(prompt_id, model, configuration)
    started = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "run_type": "experiment",
        "schema_version": SCHEMA_VERSION,
        "generation_key": key,
        "prompt_id": prompt_id,
        "prompt": selected_prompt,
        "model": model,
        "configuration": configuration.name,
        "temperature": configuration.temperature,
        "top_k": configuration.top_k,
        "top_p": configuration.top_p,
        "num_predict": num_predict,
        "started_at": started.isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "status": "error",
        "retry_attempts": 0,
        "retry_errors": [],
        "error": None,
        "raw_response": None,
        "ollama_response": None,
        "request_options": {
            "temperature": configuration.temperature,
            "top_k": configuration.top_k,
            "top_p": configuration.top_p,
        },
    }
    request_payload = {
        "model": model,
        "prompt": selected_prompt,
        "stream": False,
        "options": {
            "temperature": configuration.temperature,
            "top_k": configuration.top_k,
            "top_p": configuration.top_p,
        },
    }
    if num_predict is not None:
        request_payload["options"]["num_predict"] = num_predict
    request_started = time.perf_counter()
    try:
        for attempt in range(MAX_TRANSIENT_RETRIES + 1):
            try:
                response = session.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=request_payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                ollama_response = response.json()
                raw_response = ollama_response.get("response")
                if not isinstance(raw_response, str):
                    raise ValueError("Ollama response did not contain a string response.")
                result["status"] = "success"
                result["raw_response"] = raw_response
                result["ollama_response"] = ollama_response
                break
            except (requests.RequestException, ValueError) as error:
                if attempt >= MAX_TRANSIENT_RETRIES or not is_transient_error(error):
                    result["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    break
                result["retry_attempts"] = attempt + 1
                result["retry_errors"].append(
                    {"type": type(error).__name__, "message": str(error)}
                )
                print(
                    f"Transient failure for {key}; retry "
                    f"{attempt + 1}/{MAX_TRANSIENT_RETRIES}.",
                    flush=True,
                )
    finally:
        result["completed_at"] = utc_timestamp()
        result["duration_seconds"] = time.perf_counter() - request_started
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    """Create a result file exclusively so an existing result is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")


def run_test_generation() -> Path:
    """Execute exactly one fixed test generation and persist its complete record."""
    prompt = PROMPTS_BY_ID[TEST_PROMPT_ID]
    started = datetime.now(timezone.utc)
    started_at = started.isoformat()
    output_path = result_path(started)
    result: dict[str, Any] = {
        "run_type": "test",
        "prompt_id": TEST_PROMPT_ID,
        "prompt": prompt.text,
        "model": TEST_MODEL,
        "temperature": TEST_CONFIG.temperature,
        "top_k": TEST_CONFIG.top_k,
        "top_p": TEST_CONFIG.top_p,
        "started_at": started_at,
        "completed_at": None,
        "duration_seconds": None,
        "status": "error",
        "error": None,
        "raw_response": None,
        "ollama_response": None,
    }

    try:
        if TEST_MODEL not in MODEL_TAGS:
            raise ValueError(f"Model is not in the configured model tags: {TEST_MODEL}")
        with requests.Session() as session:
            verify_ollama_and_model(session)
            request_payload = {
                "model": TEST_MODEL,
                "prompt": prompt.text,
                "stream": False,
                "options": {
                    "temperature": TEST_CONFIG.temperature,
                    "top_k": TEST_CONFIG.top_k,
                    "top_p": TEST_CONFIG.top_p,
                },
            }
            request_started = time.perf_counter()
            response = session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=request_payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            duration_seconds = time.perf_counter() - request_started
            response.raise_for_status()
            ollama_response = response.json()
            raw_response = ollama_response.get("response")
            if not isinstance(raw_response, str):
                raise ValueError("Ollama response did not contain a string response.")
            result["status"] = "success"
            result["raw_response"] = raw_response
            result["ollama_response"] = ollama_response
            result["duration_seconds"] = duration_seconds
    except (requests.RequestException, ValueError, RuntimeError) as error:
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        result["completed_at"] = utc_timestamp()
        if result["duration_seconds"] is None:
            result["duration_seconds"] = (
                datetime.fromisoformat(result["completed_at"]).timestamp()
                - started.timestamp()
            )
        write_result(output_path, result)

    if result["status"] != "success":
        raise RuntimeError(result["error"]["message"])
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run exactly one fixed D1/qwen2.5:3b test generation.",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run or resume all 36 revised experiment generations.",
    )
    return parser.parse_args()


def run_plan(
    planned: list[tuple[str, str, DecodingConfig]],
    results_path: Path,
    label: str,
    num_predict: int | None,
    prompt_texts: dict[str, str] | None = None,
) -> int:
    """Run every missing generation in a plan sequentially."""
    successful_keys = load_successful_keys(results_path)
    skipped = sum(
        generation_key(prompt_id, model, configuration) in successful_keys
        for prompt_id, model, configuration in planned
    )
    completed = skipped
    failures = 0
    print(
        f"{label} planned: {len(planned)}; skipped successful: {skipped}; "
        f"remaining: {len(planned) - skipped}",
        flush=True,
    )
    with requests.Session() as session:
        ollama_version = verify_ollama_and_model(session)
        for prompt_id, model, configuration in planned:
            key = generation_key(prompt_id, model, configuration)
            if key in successful_keys:
                continue
            result = execute_generation(
                session,
                prompt_id,
                model,
                configuration,
                prompt_text=prompt_texts.get(prompt_id) if prompt_texts else None,
                num_predict=num_predict,
            )
            result["ollama_version"] = ollama_version
            append_jsonl(results_path, result)
            completed += 1
            if result["status"] != "success":
                failures += 1
            print(
                f"Completed: {completed}/{len(planned)}; "
                f"failed: {failures}; remaining: {len(planned) - completed}; "
                f"status: {result['status']}; key: {key}",
                flush=True,
            )
    return 0 if failures == 0 else 1


def run_experiment() -> int:
    """Run every missing revised generation sequentially."""
    planned = planned_generations()
    if len(planned) != PLANNED_GENERATION_COUNT:
        raise RuntimeError("The revised planned generation count does not equal 36.")
    return run_plan(planned, REVISED_RESULTS_PATH, "Revised experiment", None)


def main() -> int:
    args = parse_args()
    selected_modes = sum(
        bool(mode) for mode in (args.test, args.experiment)
    )
    if selected_modes != 1:
        print("Choose exactly one run mode.", file=sys.stderr)
        return 2
    if args.test:
        try:
            path = run_test_generation()
        except RuntimeError as error:
            print(f"Test generation failed: {error}", file=sys.stderr)
            return 1
        print(f"Test generation succeeded; saved to {path}")
        return 0
    try:
        return run_experiment()
    except RuntimeError as error:
        print(f"Full experiment failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
