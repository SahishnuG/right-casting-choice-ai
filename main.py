#!/usr/bin/env python
import sys
import os
import json
import warnings
from typing import Any, Dict, List, Optional

# Ensure project imports work when running this script from repo root
from pathlib import Path
sys.path.append(str((Path(__file__).parent / "src").resolve()))

# Crew and tools
from right_casting_choice_ai.crew import RightCastingChoiceAi

# Optional dotenv
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():  # type: ignore
        return None


def normalize_result(result: Any) -> Dict[str, Any]:
    """
    Convert a crew result object into a JSON-serializable dict in a robust way.
    Handles objects exposing .to_dict(), .raw, .results, .tasks_output, or being mappable to dict().
    """
    # Prefer to operate on a fresh dict
    result_dict: Dict[str, Any] = {}

    try:
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict() or {}
        elif hasattr(result, "raw"):
            # some crew libs use .raw
            raw = result.raw
            # If raw is string, attempt json.loads
            if isinstance(raw, str):
                try:
                    result_dict = json.loads(raw)
                except Exception:
                    result_dict = {"raw": raw}
            elif isinstance(raw, dict):
                result_dict = raw.copy()
            else:
                result_dict = {"raw": str(raw)}
        # fallback: result.results (some SDKs)
        if not result_dict and hasattr(result, "results") and isinstance(getattr(result, "results"), dict):
            result_dict = getattr(result, "results").copy()
        # fallback: tasks_output list like objects
        if not result_dict and hasattr(result, "tasks_output"):
            try:
                tasks_out = []
                for t in getattr(result, "tasks_output", []):
                    name = getattr(t, "name", None) or getattr(t, "task_name", None)
                    output = getattr(t, "output", None) or getattr(t, "result", None)
                    raw_output = getattr(t, "raw_output", None)
                    # Try to coerce JSON strings into objects
                    def _coerce(x):
                        if isinstance(x, str):
                            try:
                                return json.loads(x)
                            except Exception:
                                return x
                        return x
                    tasks_out.append({
                        "name": name,
                        "output": _coerce(output),
                        "raw_output": _coerce(raw_output),
                    })
                if tasks_out:
                    result_dict = {"tasks": tasks_out}
            except Exception:
                pass
        # last resort: try to cast to dict
        if not result_dict:
            try:
                result_dict = dict(result)
            except Exception:
                # As a final fallback, return repr
                result_dict = {"repr": repr(result)}
    except Exception as e:
        # If anything unexpected happens during normalization, return a minimal structure
        result_dict = {"error": f"normalize_result failed: {e}", "repr": repr(result)}

    return result_dict


def safe_kickoff(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create crew and run a single kickoff, returning a normalized dict.
    """
    try:
        crew = RightCastingChoiceAi().crew()
        result = crew.kickoff(inputs=inputs)
        return normalize_result(result)
    except Exception as e:
        return {"error": f"Crew kickoff failed: {e}", "exception": repr(e)}


def print_summary_and_json(result_dict: Dict[str, Any]):
    """
    Print a concise human-readable summary followed by the full JSON result.
    """
    try:
        # Try common shapes
        chars = result_dict.get("extract_characters_task") or result_dict.get("extract_characters") or []
        sims = result_dict.get("similar_movies_task") or result_dict.get("similar_movies") or []
        ranking = result_dict.get("rank_candidates_task") or result_dict.get("rank_candidates") or result_dict.get("ranking") or {}

        # fallback: search within tasks list
        if not sims and isinstance(result_dict.get("tasks"), list):
            for t in result_dict["tasks"]:
                if isinstance(t, dict) and t.get("name") in ("similar_movies_task", "similar_movies_and_omdb", "similar_movies"):
                    sims = t.get("output") or t.get("result") or t.get("raw_output") or []

        if not ranking and isinstance(result_dict.get("tasks"), list):
            for t in result_dict["tasks"]:
                if isinstance(t, dict) and t.get("name") in ("rank_candidates_task", "rank_candidates", "budget_ranker"):
                    out = t.get("output") or t.get("result") or t.get("raw_output") or {}
                    if isinstance(out, (dict, list)):
                        ranking = out

        # Print counts
        print("=== Casting Crew Run Summary ===")
        print(f"Characters found: {len(chars) if isinstance(chars, (list, dict)) else 'unknown'}")
        print(f"Similar movies fetched: {len(sims) if isinstance(sims, list) else 'unknown'}")

        cand_count = 0
        if isinstance(ranking, dict):
            cand_count = len(ranking.get("candidates", []))
        elif isinstance(ranking, list):
            cand_count = len(ranking)
        print(f"Candidates ranked: {cand_count}")

        # Print per-role candidate suggestions
        cand_list = []
        if isinstance(ranking, dict):
            cand_list = ranking.get("candidates", []) or []
        elif isinstance(ranking, list):
            cand_list = ranking

        top_names = []
        for row in (cand_list or [])[:6]:
            if isinstance(row, dict):
                name = row.get("name") or row.get("actor") or row.get("actor_name")
                if name:
                    top_names.append(name)
        if not top_names:
            top_names = []

        # Ensure chars is a list to iterate
        if not isinstance(chars, list):
            # if it's a dict with roles, try to coerce
            chars = [chars] if chars else [{"role": "Lead"}]

        print("\n=== Suggested Actors per Role ===")
        for ch in chars:
            if isinstance(ch, dict):
                role = ch.get("role") or ch.get("name") or "Role"
            else:
                role = str(ch)
            print(f"{role}: {', '.join(top_names) if top_names else 'No candidates'}")

        # Full JSON
        print("\n=== Full JSON Result ===")
        try:
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))
        except Exception:
            print(str(result_dict))
    except Exception as e:
        print(f"Error printing summary: {e}")
        print("Full result (repr):", repr(result_dict))


def run():
    """
    Run the casting crew with a sample plot and parameters.
    Prints a concise summary and JSON result to stdout.
    """
    # Sample inputs aligned to the casting workflow
    inputs = {
        "plot": (
            "A righteous and fearless police officer takes on corruption and crime "
            "to restore justice in his city."
        ),
        "n_similar": 3,
        "usd_to_inr": 83.0,
        "user_budget_inr": 100_000_000,
        "industry": "hollywood",  # use "bollywood" to prefer Indian actors/movies
    }

    # Propagate env for tools if set in system
    try:
        load_dotenv()
    except Exception:
        pass

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if gemini_key:
        os.environ.setdefault("GEMINI_API_KEY", gemini_key)
        os.environ.setdefault("GOOGLE_API_KEY", gemini_key)

    serper_key = os.getenv("SERPER_API_KEY") or os.getenv("SERPERDEV_API_KEY") or ""
    if serper_key:
        os.environ.setdefault("SERPER_API_KEY", serper_key)

    omdb_key = os.getenv("OMDB_API_KEY") or ""
    if omdb_key:
        os.environ.setdefault("OMDB_API_KEY", omdb_key)

    # Run kickoff and print results
    result_dict = safe_kickoff(inputs)
    print_summary_and_json(result_dict)


def train():
    """
    Train the crew for a given number of iterations and save to filename.
    Usage: python main.py train <n_iterations> <filename>
    """
    if len(sys.argv) < 4:
        raise SystemExit("Usage: train <n_iterations> <filename>")

    try:
        n_iter = int(sys.argv[2])
        filename = sys.argv[3]
    except Exception as e:
        raise SystemExit(f"Invalid args for train: {e}")

    inputs = {
        "plot": "A fearless officer fights corruption.",
        "n_similar": 2,
        "usd_to_inr": 83.0,
        "user_budget_inr": 50_000_000,
    }

    try:
        RightCastingChoiceAi().crew().train(n_iterations=n_iter, filename=filename, inputs=inputs)
        print(f"Training complete — saved to {filename}")
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    Usage: python main.py replay <task_id>
    """
    if len(sys.argv) < 3:
        raise SystemExit("Usage: replay <task_id>")
    task_id = sys.argv[2]
    try:
        RightCastingChoiceAi().crew().replay(task_id=task_id)
        print(f"Replayed task {task_id}")
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    Usage: python main.py test <n_iterations> <eval_llm>
    """
    if len(sys.argv) < 4:
        raise SystemExit("Usage: test <n_iterations> <eval_llm>")

    try:
        n_iter = int(sys.argv[2])
        eval_llm = sys.argv[3]
    except Exception as e:
        raise SystemExit(f"Invalid args for test: {e}")

    inputs = {
        "plot": "A fearless officer fights corruption.",
        "n_similar": 2,
        "usd_to_inr": 83.0,
        "user_budget_inr": 50_000_000,
    }

    try:
        RightCastingChoiceAi().crew().test(n_iterations=n_iter, eval_llm=eval_llm, inputs=inputs)
        print("Test run complete")
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <run|train|replay|test> [args...]")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "run":
        run()
    elif cmd == "train":
        train()
    elif cmd == "replay":
        replay()
    elif cmd == "test":
        test()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python main.py <run|train|replay|test> [args...]")
        sys.exit(2)
