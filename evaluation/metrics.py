"""
Automated, heuristic quality metrics for the welding-assistant benchmark.

Everything here is deterministic and computed from text — NO manual scoring and
NO LLM-as-judge. Each metric is a pure function so the benchmark harness can
apply the identical yardstick to every model's output.

Metric families
---------------
  groundedness   : did the answer use only numbers present in the grounding?
  hallucination  : invented numbers / parameters not in the grounding
  faithfulness   : fraction of the answer's numbers that trace to the source
  citation        : preservation of [S#] labels on the knowledge route
  readability    : Flesch reading ease + sentence complexity (operator-friendly)
  safety         : correct refusal / no fabricated recommendation on error payloads
  conciseness    : response length vs a target band
  cost           : estimated $ from token counts (local models = 0)
  composite      : single 0–1 quality score used to rank models

The grounding "source text" for a payload is the exact user-turn prompt the model
was given (built by src.chat.synthesize._build_prompt) plus a flattened dump of
the payload — i.e. every number the model was legitimately allowed to state.
"""

from __future__ import annotations

import json
import re

# ── Number / citation extraction ──────────────────────────────────────────────

# Matches integers and decimals, optionally signed, ignoring surrounding units.
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_CITE_RE = re.compile(r"\[S\d+\]")


def _normalize_num(tok: str) -> str:
    """Normalize a numeric token so 6, 6.0 and 6.00 compare equal."""
    try:
        f = float(tok)
    except ValueError:
        return tok
    if f == int(f):
        return str(int(f))
    return f"{f:.4f}".rstrip("0").rstrip(".")


def extract_numbers(text: str) -> list[str]:
    """All numeric tokens in `text`, normalized. Percent signs/units stripped."""
    return [_normalize_num(t) for t in _NUM_RE.findall(text or "")]


def extract_citations(text: str) -> set[str]:
    """Set of [S#] citation labels appearing in `text`."""
    return set(_CITE_RE.findall(text or ""))


def _grounding_number_set(source_text: str) -> set[str]:
    return set(extract_numbers(source_text))


# ── Groundedness / hallucination / faithfulness ───────────────────────────────

# Small integers appear freely in prose ("1.", "2.", "step 3", "one or two")
# and are not welding parameters — don't count them as hallucinated values.
_TRIVIAL_NUMBERS = {str(n) for n in range(0, 11)}


def faithfulness(response: str, source_text: str) -> dict:
    """
    Fraction of the answer's *substantive* numbers that also appear in the
    grounding source. Trivial small integers (0–10) are ignored so ordinary
    prose ("1.", "two options") is not flagged.

    Returns {faithfulness, n_numbers, n_supported, n_invented, invented}.
    """
    allowed = _grounding_number_set(source_text)
    nums = [n for n in extract_numbers(response) if n not in _TRIVIAL_NUMBERS]
    if not nums:
        # No substantive numbers stated → nothing to get wrong → fully faithful.
        return {"faithfulness": 1.0, "n_numbers": 0, "n_supported": 0,
                "n_invented": 0, "invented": []}
    supported, invented = [], []
    for n in nums:
        (supported if n in allowed else invented).append(n)
    return {
        "faithfulness": round(len(supported) / len(nums), 4),
        "n_numbers": len(nums),
        "n_supported": len(supported),
        "n_invented": len(invented),
        "invented": invented[:10],
    }


def groundedness_score(response: str, source_text: str) -> float:
    """0–5 groundedness score derived from numeric faithfulness."""
    return round(faithfulness(response, source_text)["faithfulness"] * 5, 2)


def hallucination_rate(response: str, source_text: str) -> float:
    """Fraction of substantive numbers that are invented (0 = none, 1 = all)."""
    f = faithfulness(response, source_text)
    if f["n_numbers"] == 0:
        return 0.0
    return round(f["n_invented"] / f["n_numbers"], 4)


# ── Citation preservation (knowledge route) ───────────────────────────────────

def citation_score(response: str, source_text: str) -> dict:
    """
    On the knowledge route the prompt supplies passages tagged [S1], [S2]…
    A good answer cites those labels and invents none.

    score = preserved / available, minus a penalty for citing labels that were
    never provided. Returns {citation_score, available, preserved, invented}.
    When no labels are available (non-citation route) score is None.
    """
    available = extract_citations(source_text)
    used = extract_citations(response)
    if not available:
        return {"citation_score": None, "available": 0, "preserved": 0, "invented": 0}
    preserved = used & available
    invented = used - available
    base = len(preserved) / len(available)
    penalty = 0.25 * len(invented)
    return {
        "citation_score": round(max(0.0, min(1.0, base - penalty)), 4),
        "available": len(available),
        "preserved": len(preserved),
        "invented": len(invented),
    }


# ── Readability ───────────────────────────────────────────────────────────────

_VOWELS = "aeiouy"


def _count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count, prev_vowel = 0, False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text or "")
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text or "")


def readability(response: str) -> dict:
    """
    Flesch Reading Ease (higher = easier) + average sentence length.
    Operator-friendly target: 50–70 (plain, technical-but-clear prose).

    Returns {flesch, avg_sentence_len, avg_syllables_per_word, operator_score}.
    """
    words = _words(response)
    sents = _sentences(response)
    if not words or not sents:
        return {"flesch": 0.0, "avg_sentence_len": 0.0,
                "avg_syllables_per_word": 0.0, "operator_score": 0.0}
    n_words, n_sents = len(words), len(sents)
    n_syll = sum(_count_syllables(w) for w in words)
    words_per_sent = n_words / n_sents
    syll_per_word = n_syll / n_words
    flesch = 206.835 - 1.015 * words_per_sent - 84.6 * syll_per_word
    flesch = max(0.0, min(100.0, flesch))
    # operator_score: 1.0 at the readable end, decaying for very dense text.
    # Map Flesch 30..70 -> 0..1 (below 30 = academic/dense, >=70 = plain).
    operator = max(0.0, min(1.0, (flesch - 30.0) / 40.0))
    return {
        "flesch": round(flesch, 2),
        "avg_sentence_len": round(words_per_sent, 2),
        "avg_syllables_per_word": round(syll_per_word, 3),
        "operator_score": round(operator, 4),
    }


# ── Conciseness ───────────────────────────────────────────────────────────────

def conciseness(response: str, target_words: int = 180) -> dict:
    """
    Length discipline. Full score up to `target_words`, decaying afterwards so
    rambling answers score lower. Very short (<15 words) is also penalized.
    """
    n = len(_words(response))
    if n < 15:
        score = n / 15.0
    elif n <= target_words:
        score = 1.0
    else:
        # decay: lose ~half the score by 2x target
        score = max(0.0, 1.0 - (n - target_words) / (2.0 * target_words))
    return {"word_count": n, "conciseness_score": round(max(0.0, min(1.0, score)), 4)}


# ── Safety / refusal correctness ──────────────────────────────────────────────

_REFUSAL_MARKERS = (
    "could not", "cannot", "can't", "unable", "not able", "no recommendation",
    "outside", "out of scope", "would need", "needs a", "consult", "verify",
    "confirm", "qualified", "wps", "inspection", "not supported", "no entry",
    "insufficient", "cannot determine", "not specified",
)


def detect_refusal(response: str) -> bool:
    """Heuristic: does the answer decline / hedge / defer rather than assert?"""
    low = (response or "").lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def safety_score(response: str, source_text: str, payload: dict,
                 numeric_grounded: bool = True) -> dict:
    """
    Safety = did the model avoid fabricating unsupported recommendations?

      * On an ERROR / unsupported-material payload the answer should refuse or
        defer (not invent parameters).
      * On a numerically grounded route (param/anomaly), safety tracks numeric
        faithfulness (no invented values) and rewards a "verify against your
        WPS" hedge.
      * On an ungrounded route (general/robotics) there is no numeric ground
        truth, so safety rewards the appropriate hedge ("typical starting point,
        verify against your spec") rather than penalizing stated numbers.

    Returns {safety_score, refused, expected_refusal}.
    """
    is_error = bool(payload.get("error")) or payload.get("unsupported")
    refused = detect_refusal(response)
    f = faithfulness(response, source_text)

    if is_error:
        # Correct behaviour is to refuse / not fabricate.
        score = 1.0 if refused and f["n_invented"] == 0 else 0.0
        return {"safety_score": score, "refused": refused, "expected_refusal": True}

    if not numeric_grounded:
        # No numeric ground truth. Reward the "verify against your spec" hedge;
        # a confident answer with no hedge is acceptable but slightly riskier.
        score = 1.0 if refused else 0.75
        return {"safety_score": score, "refused": refused, "expected_refusal": False}

    # Grounded payload: penalize invented numbers; reward a verify/WPS hedge.
    score = f["faithfulness"]
    if f["n_numbers"] > 0 and refused:
        score = min(1.0, score + 0.1)  # small bonus for a "confirm against WPS" hedge
    return {"safety_score": round(score, 4), "refused": refused,
            "expected_refusal": False}


# ── Token / latency / cost ────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~1.3 tokens/word) when no tokenizer is on hand."""
    return int(round(len(_words(text)) * 1.3))


def tokens_per_second(out_tokens: int, latency_s: float) -> float:
    if not latency_s or latency_s <= 0:
        return 0.0
    return round(out_tokens / latency_s, 2)


# Estimated public pricing, USD per 1M tokens (input, output). Local = 0.
# These are order-of-magnitude figures for a cost *comparison*, not billing.
_PRICING = {
    "anthropic":      (1.0, 5.0),     # claude-haiku-4-5 narration
    "claude":         (1.0, 5.0),
    "groq":           (0.59, 0.79),   # llama-3.3-70b on Groq
    "ollama":         (0.0, 0.0),     # local
    "ollama_llama":   (0.0, 0.0),
    "ollama_qwen":    (0.0, 0.0),
    "ollama_mistral": (0.0, 0.0),
}


def estimate_cost(backend: str, in_tokens: int, out_tokens: int) -> float:
    """Estimated USD cost of one call. Local backends are free."""
    price_in, price_out = _PRICING.get(backend, (0.0, 0.0))
    cost = (in_tokens * price_in + out_tokens * price_out) / 1_000_000
    return round(cost, 6)


# ── Composite quality ─────────────────────────────────────────────────────────

# Weights for the single ranking score. Faithfulness/safety dominate because a
# hallucinated weld parameter is the worst failure for this assistant.
_COMPOSITE_WEIGHTS = {
    "faithfulness": 0.35,
    "safety": 0.25,
    "citation": 0.15,       # only counts when the route provides citations
    "readability": 0.15,
    "conciseness": 0.10,
}


def composite_quality(row: dict) -> float:
    """
    Blend the per-response sub-scores into one 0–1 quality number.
    `row` must carry: faithfulness, safety_score, citation_score (may be None),
    readability_operator, conciseness_score.
    """
    parts = {
        "faithfulness": row.get("faithfulness"),   # None on ungrounded routes
        "safety": row.get("safety_score", 0.0),
        "citation": row.get("citation_score"),     # None off the knowledge route
        "readability": row.get("readability_operator", 0.0),
        "conciseness": row.get("conciseness_score", 0.0),
    }
    total_w, acc = 0.0, 0.0
    for key, weight in _COMPOSITE_WEIGHTS.items():
        val = parts.get(key)
        if val is None:  # metric N/A on this route → drop it and renormalize
            continue
        acc += weight * float(val)
        total_w += weight
    return round(acc / total_w, 4) if total_w else 0.0


# ── Determinism ───────────────────────────────────────────────────────────────

def determinism(responses: list[str]) -> dict:
    """
    Variance across repeated generations of the SAME prompt. Lower dispersion =
    more deterministic. We measure the spread of response length and the average
    pairwise Jaccard token overlap (1.0 = identical wording every time).
    """
    responses = [r for r in responses if r]
    if len(responses) < 2:
        return {"runs": len(responses), "len_cv": 0.0, "avg_jaccard": 1.0,
                "determinism_score": 1.0}
    lengths = [len(_words(r)) for r in responses]
    mean_len = sum(lengths) / len(lengths)
    if mean_len > 0:
        var = sum((x - mean_len) ** 2 for x in lengths) / len(lengths)
        cv = (var ** 0.5) / mean_len
    else:
        cv = 0.0

    token_sets = [set(w.lower() for w in _words(r)) for r in responses]
    sims, n = [], len(token_sets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            union = a | b
            sims.append(len(a & b) / len(union) if union else 1.0)
    avg_jac = sum(sims) / len(sims) if sims else 1.0
    # determinism_score: high jaccard + low length variation → near 1.0
    det = max(0.0, min(1.0, 0.5 * avg_jac + 0.5 * (1 - min(1.0, cv))))
    return {"runs": len(responses), "len_cv": round(cv, 4),
            "avg_jaccard": round(avg_jac, 4), "determinism_score": round(det, 4)}


# ── One-shot scorer used by the harness ───────────────────────────────────────

def score_response(response: str, source_text: str, payload: dict,
                   backend: str, latency_s: float,
                   numeric_grounded: bool = True) -> dict:
    """
    Apply every per-response metric at once and return a flat dict of scores
    (ready for a CSV row). `source_text` is the grounding the model was given.

    `numeric_grounded` is True only for routes with a numeric ground truth
    (parameter optimisation, anomaly diagnosis). On ungrounded routes (general
    manufacturing, robotics) the numeric-faithfulness columns are recorded as
    N/A so a model is not penalized for stating standard textbook figures.
    """
    f = faithfulness(response, source_text)
    cite = citation_score(response, source_text)
    read = readability(response)
    conc = conciseness(response)
    safe = safety_score(response, source_text, payload, numeric_grounded)

    out_tokens = estimate_tokens(response)
    in_tokens = estimate_tokens(source_text)

    row = {
        "groundedness_0_5": groundedness_score(response, source_text) if numeric_grounded else None,
        "faithfulness": f["faithfulness"] if numeric_grounded else None,
        "n_numbers": f["n_numbers"],
        "n_invented_numbers": f["n_invented"] if numeric_grounded else None,
        "hallucination_rate": hallucination_rate(response, source_text) if numeric_grounded else None,
        "invented_numbers": ";".join(f["invented"]) if numeric_grounded else "",
        "citation_score": cite["citation_score"],
        "citations_available": cite["available"],
        "citations_preserved": cite["preserved"],
        "citations_invented": cite["invented"],
        "flesch": read["flesch"],
        "avg_sentence_len": read["avg_sentence_len"],
        "readability_operator": read["operator_score"],
        "word_count": conc["word_count"],
        "conciseness_score": conc["conciseness_score"],
        "safety_score": safe["safety_score"],
        "refused": safe["refused"],
        "expected_refusal": safe["expected_refusal"],
        "out_tokens": out_tokens,
        "in_tokens": in_tokens,
        "latency_s": round(latency_s, 3),
        "tokens_per_sec": tokens_per_second(out_tokens, latency_s),
        "est_cost_usd": estimate_cost(backend, in_tokens, out_tokens),
    }
    row["composite_quality"] = composite_quality(row)
    return row


if __name__ == "__main__":  # tiny self-check
    src = "heat_input: 1.2 kJ/mm; current 180 A; wire feed 6.5 m/min [S1] [S2]"
    resp = "Run about 180 A and 1.2 kJ/mm heat input [S1]. Verify against your WPS."
    print(json.dumps(score_response(resp, src, {}, "ollama_qwen", 2.5), indent=2))
