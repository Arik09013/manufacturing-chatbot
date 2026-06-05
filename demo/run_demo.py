"""
T12 integration demo: runs the definition-of-done scenario end-to-end.

Query: "Why did line 3 slow down at 14:00?"
Expected: anomaly detected + cause + recommendation + SHAP + confidence
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.pipeline import run_pipeline
from src.chat.synthesize import _fallback_text  # no LLM key needed for demo


def run():
    queries = [
        "Why did line 3 slow down at 14:00?",
        "Why is machine_1 running hot at 15:30?",
        "Is machine_2 running normally?",
    ]

    for q in queries:
        print("=" * 60)
        print(f"QUERY: {q}")
        print("-" * 60)
        result = run_pipeline(q)

        print(f"Machine:    {result['machine_id']}")
        print(f"Window:     {result['window_start'][:16]} - {result['window_end'][:16]}")
        print(f"Anomaly:    {result['is_anomaly']}  (prob={result['anomaly_prob']:.0%})")
        print(f"Type:       {result['anomaly_type'] or 'none'}")
        conf = result["confidence"]
        print(f"Confidence: {conf['band'].upper()} ({conf['score']:.0%})")

        if result["is_anomaly"]:
            print("Causes:")
            for c in result["causes"]:
                print(f"  {c['rank']}. {c['cause']}  [{c['evidence_strength']}]")
            rec = result["recommendation"]
            print(f"Action [{rec['urgency']}]: {rec['primary']}")

        print("SHAP drivers:")
        for d in result["shap_drivers"][:3]:
            print(f"  {d['feature']}: {d['shap']:+.3f} ({d['direction'].replace('_',' ')})")

        print("\nSynthesis (fallback — set ANTHROPIC_API_KEY for LLM answer):")
        print(_fallback_text(result))
        print()


if __name__ == "__main__":
    run()
