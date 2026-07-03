"""
Comparative benchmark harness for the welding assistant.

Runs the SAME set of operator prompts through every configured LLM backend
(Claude, Llama/Groq, Qwen, Mistral) and scores each response with the automated,
heuristic metrics in evaluation.metrics — groundedness, hallucination,
faithfulness, citation preservation, readability, safety, conciseness, latency,
tokens/sec, cost, and determinism.

Design
------
* The deterministic pipeline is run ONCE per prompt to build the grounding
  payload; that payload is then narrated by each backend so every model sees an
  identical prompt (a fair comparison of *synthesis* quality).
* Nothing here crashes: a backend that is down / not pulled falls back to the
  deterministic grounded summary, and that is recorded (`fell_back`) so the
  report stays honest.

Usage
-----
    python evaluation/benchmark.py                       # all backends, full set
    python evaluation/benchmark.py --backends ollama_qwen ollama_mistral
    python evaluation/benchmark.py --limit 10            # quick smoke run
    python evaluation/benchmark.py --determinism-runs 3  # repeat prompts N times
    python evaluation/benchmark.py --no-figures --no-report

Outputs (evaluation/results/)
    <backend>_results.csv      per-prompt scores for each backend
    comparison.csv             per-backend aggregate means
    raw_outputs.json           every prompt/response/score (full record)
    figures/*.png              matplotlib comparison charts
    ../report.md               thesis-ready markdown report
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation import metrics  # noqa: E402
from src.chat.backends import backend_model, load_llm_config, probe_backend  # noqa: E402
from src.chat.synthesize import _build_prompt, _fallback_text, generate_response  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("benchmark")

_RESULTS_DIR = Path(__file__).parent / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"
_DATASET = Path(__file__).parent / "datasets" / "benchmark_set.json"

# Routes whose answers carry a numeric ground truth (numbers MUST trace to the
# computed payload). Knowledge/general/robotics answers give textbook starting
# points, so numeric faithfulness is recorded as N/A for them.
_GROUNDED_ROUTES = {"param", "anomaly"}

_DEFAULT_BACKENDS = ["anthropic", "groq", "ollama_llama", "ollama_qwen", "ollama_mistral"]


# ── Payload construction (run the deterministic pipeline once per prompt) ──────

def build_payload(prompt: str) -> tuple[dict, str, str | None]:
    """
    Route a prompt exactly like the API and run the deterministic pipeline to get
    the grounding payload. Returns (payload, route, error). Out-of-scope prompts
    become a refusal-style payload so the harness can score the refusal.
    """
    from src.api.pipeline import (
        route_question, run_pipeline, run_param_pipeline,
        run_knowledge_pipeline, run_general_pipeline,
    )
    from src.chat.intent import OutOfScopeError

    try:
        route = route_question(prompt)
        if route == "param":
            payload = run_param_pipeline(prompt)
        elif route == "knowledge":
            payload = run_knowledge_pipeline(prompt)
        elif route == "general":
            payload = run_general_pipeline(prompt)
        else:
            payload = run_pipeline(question=prompt)
        # run_knowledge_pipeline may re-route a stray general question.
        if payload.get("pipeline_type") == "general_manufacturing":
            route = "general"
        return payload, route, None
    except OutOfScopeError as exc:
        return ({"pipeline_type": "out_of_scope", "error": str(exc),
                 "unsupported": True, "question": prompt},
                "refusal", str(exc))
    except Exception as exc:
        logger.warning("payload build failed for %r: %s", prompt[:60], exc)
        return ({"pipeline_type": "error", "error": str(exc), "question": prompt},
                "error", str(exc))


def grounding_source(payload: dict) -> str:
    """
    The full text the model was legitimately grounded in: the built user prompt
    (which contains every allowed number and any [S#] passages) plus a flattened
    payload dump so numbers that live only in the payload are also credited.
    """
    try:
        user_content, _system, _max, _model = _build_prompt(payload)
    except Exception:
        user_content = ""
    return user_content + "\n" + json.dumps(payload, default=str)


# ── Core evaluation loop ───────────────────────────────────────────────────────

def build_cases(prompts: list[dict], cache_path: Path | None = None,
                rebuild: bool = False) -> list[dict]:
    """
    Build one grounding payload per prompt (the expensive deterministic pipeline
    step) and cache it. Payloads don't change between runs, so subsequent runs
    reuse the cache and skip the pipeline entirely.
    """
    import pickle
    ids = [p["id"] for p in prompts]
    if cache_path and cache_path.exists() and not rebuild:
        try:
            cached = pickle.loads(cache_path.read_bytes())
            if [c["id"] for c in cached] == ids:
                print(f"Reusing cached payloads from {cache_path.name} "
                      f"({len(cached)} prompts). Use --rebuild-payloads to refresh.")
                return cached
        except Exception:
            logger.warning("payload cache unreadable; rebuilding", exc_info=True)

    print(f"Building {len(prompts)} grounding payloads (running the pipeline once each)…")
    cases = []
    for i, item in enumerate(prompts, 1):
        payload, route, err = build_payload(item["prompt"])
        cases.append({
            "id": item["id"], "prompt": item["prompt"],
            "expected_route": item["route"], "actual_route": route,
            "payload": payload, "source": grounding_source(payload),
            "numeric_grounded": route in _GROUNDED_ROUTES,
            "build_error": err,
        })
        print(f"  [{i}/{len(prompts)}] {item['id']:<9} route={route}")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(cases))
    return cases


def evaluate(backends: list[str], prompts: list[dict], determinism_runs: int = 1,
             cache_path: Path | None = None, rebuild: bool = False) -> tuple[dict, list[dict]]:
    """
    Run every backend over every prompt and score each response.

    Returns (per_backend_rows, raw_records) where per_backend_rows maps a backend
    name to a list of flat score dicts (one per prompt).
    """
    # 1) Build (or reuse) every payload once.
    cases = build_cases(prompts, cache_path=cache_path, rebuild=rebuild)

    per_backend: dict[str, list[dict]] = {b: [] for b in backends}
    raw_records: list[dict] = []

    # 2) Narrate + score every case with each backend.
    for backend in backends:
        model = backend_model(backend)
        # Probe once so we don't hammer a down backend for every prompt.
        available, reason = probe_backend(backend)
        print(f"\n=== Backend: {backend}  (model: {model}) — {reason} ===")
        fell_back_count = 0
        for i, case in enumerate(cases, 1):
            payload = case["payload"]
            reps = max(1, determinism_runs)
            texts, latencies, first = [], [], None
            if not available:
                # Skip the network entirely: record the deterministic fallback
                # once (fast + honest). Real latencies come from a live backend.
                first = {"text": _fallback_text(payload), "fell_back": True,
                         "error": reason, "latency_s": 0.0}
                texts, latencies = [first["text"]], [0.0]
            else:
                for _ in range(reps):
                    res = generate_response(payload, backend=backend)
                    texts.append(res["text"])
                    latencies.append(res["latency_s"])
                    if first is None:
                        first = res
            fell_back_count += int(first["fell_back"])

            score = metrics.score_response(
                response=first["text"], source_text=case["source"],
                payload=payload, backend=backend,
                latency_s=min(latencies),  # best-of latency for a fair timing
                numeric_grounded=case["numeric_grounded"],
            )
            det = metrics.determinism(texts) if reps > 1 else {
                "runs": 1, "determinism_score": 1.0, "avg_jaccard": 1.0, "len_cv": 0.0}

            row = {
                "id": case["id"], "prompt": case["prompt"],
                "route": case["actual_route"], "backend": backend, "model": model,
                "fell_back": first["fell_back"], "error": first["error"] or "",
                "determinism_score": det["determinism_score"],
                "det_jaccard": det["avg_jaccard"], "det_len_cv": det["len_cv"],
                **score,
            }
            per_backend[backend].append(row)
            raw_records.append({**row, "response": first["text"],
                                "all_responses": texts if reps > 1 else None})
            print(f"  [{i}/{len(cases)}] {case['id']:<9} "
                  f"q={score['composite_quality']:.2f} "
                  f"faith={score['faithfulness']} "
                  f"lat={score['latency_s']}s "
                  f"{'(fallback)' if first['fell_back'] else ''}")
        if fell_back_count:
            print(f"  note: {fell_back_count}/{len(cases)} prompts used the "
                  f"deterministic fallback (backend unavailable).")

    return per_backend, raw_records


# ── Aggregation + output ───────────────────────────────────────────────────────

def _mean(vals: list) -> float | None:
    nums = [v for v in vals if isinstance(v, (int, float)) and v is not None]
    return round(sum(nums) / len(nums), 4) if nums else None


_NUMERIC_COLS = [
    "composite_quality", "groundedness_0_5", "faithfulness", "hallucination_rate",
    "citation_score", "flesch", "readability_operator", "conciseness_score",
    "safety_score", "word_count", "latency_s", "tokens_per_sec", "est_cost_usd",
    "determinism_score", "n_invented_numbers",
]


def aggregate(per_backend: dict[str, list[dict]]) -> list[dict]:
    """Per-backend mean of every numeric column + fallback rate."""
    agg = []
    for backend, rows in per_backend.items():
        if not rows:
            continue
        summary = {"backend": backend, "model": rows[0]["model"], "n_prompts": len(rows)}
        for col in _NUMERIC_COLS:
            summary[col] = _mean([r.get(col) for r in rows])
        summary["fallback_rate"] = round(
            sum(int(r["fell_back"]) for r in rows) / len(rows), 4)
        summary["total_cost_usd"] = round(
            sum(r.get("est_cost_usd") or 0 for r in rows), 6)
        agg.append(summary)
    agg.sort(key=lambda s: (s.get("composite_quality") or 0), reverse=True)
    return agg


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def save_outputs(per_backend: dict[str, list[dict]], agg: list[dict],
                 raw_records: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    # Per-backend CSVs (claude_results.csv, qwen_results.csv, …).
    alias = {"anthropic": "claude", "ollama_llama": "llama",
             "ollama_qwen": "qwen", "ollama_mistral": "mistral", "groq": "groq"}
    for backend, rows in per_backend.items():
        name = alias.get(backend, backend)
        _write_csv(outdir / f"{name}_results.csv", rows)
    _write_csv(outdir / "comparison.csv", agg)
    with open(outdir / "raw_outputs.json", "w", encoding="utf-8") as fh:
        json.dump(raw_records, fh, indent=2, default=str)
    print(f"\nWrote per-backend CSVs, comparison.csv and raw_outputs.json to {outdir}")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Welding-assistant LLM benchmark")
    cfg_backends = load_llm_config().get("benchmark_backends") or _DEFAULT_BACKENDS
    parser.add_argument("--backends", nargs="+", default=cfg_backends,
                        help="backend names to compare (default: from config/llm.yaml)")
    parser.add_argument("--dataset", type=Path, default=_DATASET)
    parser.add_argument("--limit", type=int, default=0,
                        help="only run the first N prompts (0 = all)")
    parser.add_argument("--determinism-runs", type=int, default=1,
                        help="repeat each prompt N times to measure variance")
    parser.add_argument("--outdir", type=Path, default=_RESULTS_DIR)
    parser.add_argument("--rebuild-payloads", action="store_true",
                        help="rebuild the grounding payload cache from scratch")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    with open(args.dataset, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    prompts = dataset["prompts"]
    if args.limit:
        prompts = prompts[: args.limit]

    print(f"Benchmarking {len(args.backends)} backends over {len(prompts)} prompts "
          f"({args.determinism_runs} run(s) each).")
    t0 = time.perf_counter()
    cache_path = args.outdir / "_payloads.pkl"
    per_backend, raw_records = evaluate(
        args.backends, prompts, args.determinism_runs,
        cache_path=cache_path, rebuild=args.rebuild_payloads)
    agg = aggregate(per_backend)
    save_outputs(per_backend, agg, raw_records, args.outdir)

    # Console leaderboard.
    print("\n" + "=" * 64)
    print("LEADERBOARD (mean composite quality, higher = better)")
    print("=" * 64)
    for s in agg:
        print(f"  {s['backend']:<16} {s['model']:<24} "
              f"quality={s['composite_quality']}  "
              f"latency={s['latency_s']}s  fallback={s['fallback_rate']}")

    if not args.no_figures:
        try:
            from evaluation import plots
            plots.generate_all(agg, per_backend, _FIGURES_DIR)
            print(f"Figures written to {_FIGURES_DIR}")
        except Exception:
            logger.warning("Figure generation failed", exc_info=True)

    if not args.no_report:
        try:
            from evaluation import report
            report.generate(agg, per_backend, raw_records, prompts,
                            Path(__file__).parent / "report.md")
            print(f"Report written to {Path(__file__).parent / 'report.md'}")
        except Exception:
            logger.warning("Report generation failed", exc_info=True)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
