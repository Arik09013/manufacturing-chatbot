"""Quick sanity test for the full pipeline (T6-T9)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.pipeline import run_pipeline

def main():
    result = run_pipeline("Why did line 1 slow down at 15:24?")
    print("=== Pipeline result ===")
    print(f"Machine:    {result['machine_id']}")
    print(f"Window:     {result['window_start']} to {result['window_end']}")
    print(f"Anomaly:    {result['is_anomaly']} (prob={result['anomaly_prob']:.0%})")
    print(f"Type:       {result['anomaly_type']}")
    print(f"Confidence: {result['confidence']}")
    print("Causes:")
    for c in result["causes"]:
        print(f"  {c['rank']}. {c['cause']} [{c['evidence_strength']}]")
    rec = result.get("recommendation", {})
    print(f"Recommendation: {rec.get('primary', 'N/A')}")
    print(f"SHAP summary:\n{result['shap_text']}")
    print()
    print("T6-T9 pipeline: PASS")

if __name__ == "__main__":
    main()
