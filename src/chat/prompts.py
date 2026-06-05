"""
Prompt templates for the LLM synthesis layer.

The LLM receives structured JSON from the deterministic pipeline and
converts it to a concise, operator-facing plain-language answer.
It does NOT detect anomalies, retrieve data, or make up facts.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an industrial AI assistant helping factory operators understand machine anomalies.

You receive structured diagnostic data from a deterministic ML pipeline and your job is to explain it clearly and concisely in plain language that an operator can act on immediately.

Rules:
- Speak directly and practically. No jargon unless necessary.
- Always mention the machine, the anomaly type, the top cause, and the recommended action.
- Always state the confidence level.
- Keep your response under 150 words.
- Do NOT invent causes or recommendations not present in the data.
- If confidence is low, say so explicitly."""


SYNTHESIS_PROMPT = """The diagnostic pipeline returned the following results for the operator's question.

QUESTION: {question}

PIPELINE RESULTS:
- Machine: {machine_id}
- Anomaly detected: {is_anomaly}
- Anomaly type: {anomaly_type}
- Anomaly probability: {anomaly_prob:.0%}
- Confidence: {confidence_band} ({confidence_score:.0%})

TOP CAUSES (ranked by evidence):
{causes_text}

RECOMMENDED ACTIONS:
- Primary: {action_primary}
- Secondary: {action_secondary}
- Urgency: {urgency}

KEY SENSOR SIGNALS (SHAP-based):
{shap_text}

Write a concise, operator-friendly explanation of what is happening and what should be done."""


NO_ANOMALY_PROMPT = """The diagnostic pipeline returned the following results for the operator's question.

QUESTION: {question}

PIPELINE RESULTS:
- Machine: {machine_id}
- Anomaly detected: No
- Anomaly probability: {anomaly_prob:.0%}
- Confidence: {confidence_band} ({confidence_score:.0%})

The machine appears to be operating within normal parameters for the queried time window.

Write a brief, reassuring plain-language response. If the operator describes symptoms that don't match the pipeline result, note the discrepancy and suggest manual inspection."""


def build_synthesis_prompt(payload: dict) -> str:
    """Build the user-turn content from a pipeline result payload."""
    if not payload.get("is_anomaly"):
        return NO_ANOMALY_PROMPT.format(
            question=payload.get("question", ""),
            machine_id=payload.get("machine_id", "unknown"),
            anomaly_prob=payload.get("anomaly_prob", 0.0),
            confidence_band=payload.get("confidence", {}).get("band", "unknown"),
            confidence_score=payload.get("confidence", {}).get("score", 0.0),
        )

    causes = payload.get("causes", [])
    causes_text = "\n".join(
        f"  {c['rank']}. {c['cause']} (evidence: {c['evidence_strength']})"
        for c in causes
    ) or "  None identified"

    drivers = payload.get("shap_drivers", [])
    shap_text = "\n".join(
        f"  - {d['feature']}: {d['shap']:+.3f} ({d['direction'].replace('_', ' ')})"
        for d in drivers[:5]
    ) or "  None available"

    rec = payload.get("recommendation", {})
    conf = payload.get("confidence", {})

    return SYNTHESIS_PROMPT.format(
        question=payload.get("question", ""),
        machine_id=payload.get("machine_id", "unknown"),
        is_anomaly=payload.get("is_anomaly", False),
        anomaly_type=payload.get("anomaly_type", "unknown"),
        anomaly_prob=payload.get("anomaly_prob", 0.0),
        confidence_band=conf.get("band", "unknown"),
        confidence_score=conf.get("score", 0.0),
        causes_text=causes_text,
        action_primary=rec.get("primary", "Inspect machine"),
        action_secondary=rec.get("secondary", "Monitor closely"),
        urgency=rec.get("urgency", "medium"),
        shap_text=shap_text,
    )
