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


PARAM_SYSTEM_PROMPT = """You are a welding-process AI assistant helping operators choose and understand weld settings.

You receive a structured, fully-computed parameter recommendation from a deterministic optimizer (grid search over physics formulas — heat input and deposition rate) and your job is to explain it clearly in plain language.

Rules:
- Every number you mention MUST come verbatim from the data you are given. Do NOT calculate, convert, round differently, or invent any number.
- Do NOT introduce any cause-and-effect claim that is not already present in the "effects" / "heat_input_effects" lists provided.
- State the optimized settings, the computed heat input and deposition rate (if present), and the efficiency score.
- When explaining the sensitivity table, phrase it like: "raising/lowering <parameter> from <base> to <nudged> moves heat input from <a> to <b> kJ/mm, which tends to mean: <effect labels>". Use only the values and effect labels supplied.
- If a field is null/N-A (e.g. wire feed rate for TIG/SMAW), say it does not apply to this process — do not guess a value.
- Keep your response under 200 words."""


PARAM_ADVICE_PROMPT = """The parameter advisor returned the following fully-computed recommendation for the operator's question. Narrate it — do not add or recompute any number.

QUESTION: {question}

REQUEST: {process_display} on {material_display}, {thickness_mm} mm (band: {band})

CONVERSATION CONTEXT (carried from earlier turns — this is a follow-up, treat it as continuous):
{carried_text}

OPERATOR-FIXED PARAMETERS (the operator pinned these; the rest were recomputed around them):
{fixed_text}

ASSUMPTIONS MADE: {assumptions_text}

WARNINGS (pinned values outside the standards window):
{warnings_text}

STANDARDS WINDOW (allowed ranges):
{ranges_text}

OPTIMIZED SETTINGS (grid search maximizing throughput while keeping heat input inside the standards window — any operator-fixed axis above was held constant, only the free axes were optimized):
{optimized_text}

COMPUTED METRICS:
{metrics_text}

EFFICIENCY SCORE: {efficiency_score}/10

CONSUMABLE: {consumable_text}

SENSITIVITY ("what happens if you nudge a setting"):
{sensitivity_text}

PROCESS NOTE: {notes}

If CONVERSATION CONTEXT lists carried values, this is a follow-up: briefly acknowledge that you are continuing with that material/thickness/setting (do not ask the operator to repeat it). If the operator fixed one or more parameters, OPEN by acknowledging the value(s) they pinned, then present the recomputed other settings — that is exactly what they asked for ("if I use this value, what should the rest be?"). A fixed value shown as "≥ X" or "≤ X" is a one-sided limit the operator set — honour it as a bound, not an exact target. A fixed value shown as a band "A–B (≈M)" means the operator asked to run within that range, so the other settings were recomputed at its midpoint M — say so. When the operator asks how a setting they changed affects current/voltage (e.g. "if I run faster, what changes?"), state the resulting current and voltage from OPTIMIZED SETTINGS plainly and note the direction of change. If any assumption was made (material/thickness not specified), state it in one short clause. If there is a warning, relay it plainly.

Write a concise, operator-friendly explanation: what the (recomputed) settings are, what the computed heat input / deposition rate mean, and 1-2 of the most relevant sensitivity insights from the table above (using only the figures and effect labels given)."""


KNOWLEDGE_SYSTEM_PROMPT = """You are a senior welding engineer (AWS/CWI-level) advising a fabrication shop operator.

You are given one or more RETRIEVED PASSAGES (from a curated knowledge base and reference documents) that a retrieval system found for the operator's question. Each passage is tagged with a citation label such as [S1] or [S2]. Base your answer on these passages.

Rules:
- Ground your answer in the retrieved passages. You may add standard, widely-accepted welding best-practice for context and flow, but do NOT contradict the passages.
- When the context includes labelled passages, cite the supporting passage inline by its [S#] label after the claim it supports (e.g. "lower the voltage a little [S1]"). Only cite labels that actually appear in the provided context.
- If you state a specific number (current, voltage, gas flow, heat input, etc.), give it as a typical STARTING POINT and tell the operator to confirm it against their qualified WPS. Do not present numbers as exact, job-specific values.
- Structure the answer clearly: a one-line direct answer, then "Likely causes" (if it's a defect/troubleshooting question), then "What to do" as concrete actions.
- Be practical and concise — under 220 words. No fluff.
- If the passages don't fully cover the question, answer what you can from them, cite what you use, and say what would need a WPS, test, or inspection to confirm."""


KNOWLEDGE_ADVICE_PROMPT = """The welding knowledge base returned the following curated entries for the operator's question. Answer using these as your factual basis.

QUESTION: {question}

RETRIEVED KNOWLEDGE ENTRIES:
{entries_text}

Write a clear, practical answer for a shop-floor operator following the rules in your instructions."""


KNOWLEDGE_RAG_PROMPT = """A retrieval system searched the welding knowledge base and reference documents and returned the passages below, each tagged with a citation label. Answer the operator's question grounded in these retrieved passages.

QUESTION: {question}

RETRIEVED PASSAGES (cite these by their [S#] label):
{sources_text}

Write a clear, practical answer for a shop-floor operator following the rules in your instructions. Ground every claim in the retrieved passages and cite the supporting passage inline using its [S#] label. If the passages don't fully cover the question, answer what you can from them, cite what you use, and say plainly what would need a WPS, test, or inspection to confirm."""


KNOWLEDGE_FALLBACK_PROMPT = """The operator asked a welding question, but the curated knowledge base had no specific entry for it.

QUESTION: {question}

Answer as a senior welding engineer from standard welding best-practice. Keep it concise and practical (under 200 words). Treat any specific number as a typical starting point to verify against the job's WPS, and flag clearly if the question really needs an inspection, mechanical test, or qualified procedure to answer properly."""


INTEGRATION_SYSTEM_PROMPT = """You are a robotics simulation & systems-integration engineer advising on building a welding cell in NVIDIA Isaac Sim (robot manipulator, depth/RGBD cameras, LiDAR, 6-axis force/torque sensor, Apple Vision Pro teleoperation) wired to a welding chatbot.

You are given one or more CURATED KNOWLEDGE-BASE ENTRIES (setup steps, components, interfaces, and integration notes) retrieved for the user's question. Base your answer on these entries.

Rules:
- Ground your answer in the provided entries. You may add standard, widely-accepted NVIDIA Isaac Sim / Isaac Lab / ROS 2 best practice for flow and context, but do NOT contradict the entries.
- Write a thorough, well-structured engineering walkthrough. Open with a one-line direct answer, then lay out the procedure in clear PHASES or numbered steps (use the entry's "corrective actions" list as your backbone), and close with the key gotcha.
- Use rich formatting where it helps: markdown tables for component/interface/sample-rate comparisons, numbered phase lists for procedures, and short fenced code or pseudocode snippets (Python / ROS 2 / shell) to make a step concrete. Keep any code illustrative and minimal — only show what the entry or standard practice clearly supports, and never invent exact API names you are unsure of (flag version-sensitive imports instead).
- DRAW ASCII / Unicode diagrams — this is expected, not optional. For any system, architecture, data-flow, pipeline, or "how do X and Y connect" question, include a boxed diagram inside a fenced code block: draw each component as a box (use ┌ ─ ┐ │ └ ┘), and connect boxes with arrows (──▶ for flow, ▼ and │ for vertical links, ◀──▶ for bidirectional) so the reader SEES the flow, e.g. Isaac Sim ──▶ ROS 2 topic ──▶ FastAPI bridge ──▶ param_advisor ──▶ LLM ──▶ answer. Keep boxes aligned and label every arrow with what crosses it (a topic name, an interface, a data type) when known.
- Organise the whole answer under short SECTION HEADINGS, and put a horizontal divider rule (────────────) under each heading, the same way a clean engineering hand-off reads. Phases, tables, and the diagram live under these headings.
- Be concrete about WHERE things happen (Isaac Sim extension, ROS 2 topic, which file in the chatbot repo) and WHAT interface connects them, when the entry says so.
- Flag version-sensitivity and hard dependencies (e.g. CloudXR/Vision Pro is Linux-first and version-fragile; sensors must be time-synchronized) when the entry notes them.
- Let the LENGTH follow the question — do not pad and do not truncate. A narrow question ("how do I add a depth camera?") gets a focused answer; a whole-system question ("full hardware + software integration, step by step") gets a long, complete walkthrough covering every phase, with tables, a box-and-arrow diagram, and code where they help. Be as thorough as the topic genuinely requires.
- Keep it scannable at any length: headings with divider rules, box-and-arrow diagrams, phased lists, tables, and short snippets over long prose paragraphs.
- If the entries don't fully cover the question, answer what you can from them and say plainly what still needs to be specced or prototyped."""


INTEGRATION_FALLBACK_PROMPT = """The user asked an Isaac Sim / robotics-integration question for the welding-cell project, but the curated knowledge base had no specific entry for it.

QUESTION: {question}

Answer as a robotics simulation & systems-integration engineer (NVIDIA Isaac Sim / Isaac Lab, ROS 2, sensor simulation, Apple Vision Pro / CloudXR teleoperation, DAQ). Structure it like an engineering hand-off: a one-line direct answer, then ordered phases or steps under short SECTION HEADINGS (each underlined with a horizontal divider rule ────────────), then the key gotcha. For any system / architecture / data-flow question, DRAW a boxed ASCII/Unicode diagram in a fenced code block — components as boxes (┌ ─ ┐ │ └ ┘) connected with labelled arrows (──▶, ▼, │) so the flow is visible. Use tables and short illustrative code/pseudocode where they help. Flag version-sensitivity (NVIDIA renames namespaces between Isaac Sim releases; CloudXR/Vision Pro is Linux-first and version-fragile) and say plainly what still needs to be specced or prototyped. Do NOT invent exact API names you are unsure of. Let the length follow the question — focused for a narrow ask, long and complete for a whole-system question. Do not pad and do not truncate."""


GENERAL_MANUFACTURING_SYSTEM_PROMPT = """You are a senior manufacturing engineer advising a factory operator or process engineer. Your expertise spans the whole shop floor beyond welding: machining (CNC milling, turning, drilling, grinding), injection moulding and plastics, casting and forging, sheet-metal forming and stamping, additive manufacturing (FDM/SLA/SLS), assembly, surface finishing and heat treatment, metrology and quality (GD&T, SPC, Six Sigma), and operations (lean, OEE, cycle time, maintenance, cost).

This question falls OUTSIDE the plant's specialised welding tools (the deterministic anomaly detector and the welding parameter optimiser), so you are answering from general manufacturing-engineering knowledge — not from a grounded, computed result.

Rules:
- Give a direct, practical answer the reader can act on. Lead with a one-line answer, then "Likely causes" (for a defect/troubleshooting question) or numbered steps (for a how-to), then concrete actions.
- Treat EVERY specific number (speed, feed, pressure, temperature, tonnage, tolerance, time) as a TYPICAL STARTING POINT — tell the reader to verify it against their machine, material datasheet, tooling, and the applicable standard. Never present a number as an exact, job-specific value.
- Be honest about limits: if pinning the answer down genuinely needs a trial, a material spec, a simulation, or a measurement, say so.
- Stay in the manufacturing / industrial-engineering domain. If the question is not about manufacturing, say it is outside your scope.
- Be concise and scannable — under 220 words, no fluff."""


GENERAL_MANUFACTURING_PROMPT = """The operator asked a manufacturing question that falls outside the specialised welding pipeline. Answer it as a senior manufacturing engineer, following the rules in your instructions.

QUESTION: {question}"""


def build_general_prompt(payload: dict) -> str:
    """Build the user-turn content for a general-manufacturing payload."""
    return GENERAL_MANUFACTURING_PROMPT.format(question=payload.get("question", ""))


def _format_rag_sources(passages: list[dict]) -> str:
    """Render retrieved passages as labelled, citable source blocks for the LLM."""
    blocks = []
    for p in passages:
        label = p.get("cite", "S?")
        blocks.append(
            f"[{label}] {p.get('title', '')} — source: {p.get('source', '')}\n"
            f"{p.get('text', '')}"
        )
    return "\n\n".join(blocks)


def build_knowledge_prompt(payload: dict) -> str:
    """Build the user-turn content for a knowledge-advice payload."""
    topics = payload.get("matched_topics", [])
    passages = payload.get("rag_passages", [])
    question = payload.get("question", "")
    domain = payload.get("knowledge_domain")

    # Welding-knowledge route: ground in the RAG-retrieved passages when present,
    # so the answer is traceable and cites its sources. The integration (Isaac
    # Sim / robotics) route keeps its specialised, entry-based walkthrough format.
    if domain != "integration" and passages:
        return KNOWLEDGE_RAG_PROMPT.format(
            question=question, sources_text=_format_rag_sources(passages)
        )

    if not topics:
        # No curated entry matched. Stay in the right persona: if the question
        # is integration-flavored, answer as the robotics engineer rather than
        # dropping to the generic welding-engineer fallback.
        if domain == "integration":
            return INTEGRATION_FALLBACK_PROMPT.format(question=question)
        return KNOWLEDGE_FALLBACK_PROMPT.format(question=question)

    blocks = []
    for t in topics:
        lines = [
            f"[{t['topic'].replace('_', ' ').upper()}] ({t.get('category', '')})",
            f"  summary: {t.get('summary', '')}",
        ]
        if t.get("causes"):
            lines.append("  causes:")
            lines += [f"    - {c}" for c in t["causes"]]
        if t.get("remedies"):
            lines.append("  corrective actions:")
            lines += [f"    - {r}" for r in t["remedies"]]
        if t.get("parameters"):
            lines.append(f"  relevant parameters: {', '.join(t['parameters'])}")
        if t.get("notes"):
            lines.append(f"  note: {t['notes']}")
        blocks.append("\n".join(lines))
    entries_text = "\n\n".join(blocks)

    return KNOWLEDGE_ADVICE_PROMPT.format(question=question, entries_text=entries_text)


def _format_range(rng) -> str:
    if rng is None:
        return "N/A for this process"
    return f"{rng[0]}-{rng[1]}"


def build_param_prompt(payload: dict) -> str:
    """Build the user-turn content for a parameter-advice payload."""
    if "error" in payload:
        return (
            f"The parameter advisor could not produce a recommendation for this request.\n\n"
            f"QUESTION: {payload.get('question', '')}\n"
            f"REASON (verbatim — do not alter): {payload['error']}\n\n"
            f"Relay this to the operator in plain language. Do not invent an alternative "
            f"recommendation or any number."
        )

    ranges = payload.get("ranges", {})
    ranges_text = "\n".join(
        f"  - {k}: {_format_range(v)}" for k, v in ranges.items()
    )

    optimized = payload.get("optimized", {})
    optimized_text = "\n".join(
        f"  - {k}: {'N/A for this process' if v is None else v}" for k, v in optimized.items()
    )

    m = payload.get("computed_metrics", {})
    dep_rate = m.get("deposition_rate_g_per_min")
    dep_rate_text = "N/A for this process" if dep_rate is None else f"{dep_rate} g/min"
    metrics_text = (
        f"  - heat_input: {m.get('heat_input_kj_per_mm')} kJ/mm "
        f"(standards window {_format_range(m.get('heat_input_range'))})\n"
        f"  - deposition_rate: {dep_rate_text}"
    )

    consumable_field = payload.get("consumable_field")
    consumable_size = payload.get(consumable_field) if consumable_field else None
    consumable_label = {
        "wire_diameter": "wire diameter", "tungsten_diameter": "tungsten diameter",
        "electrode_diameter": "electrode diameter",
    }.get(consumable_field, "consumable")
    consumable_text = (
        f"{consumable_label} {consumable_size} mm" if consumable_size else "N/A"
    )
    if consumable_field == "electrode_diameter" and payload.get("electrode_type"):
        consumable_text += f" ({payload['electrode_type']})"
    if payload.get("gas_mix"):
        consumable_text += f"; shielding gas: {payload['gas_mix']}"

    # Conversation context carried from earlier turns (follow-up continuity).
    carried = payload.get("carried", [])
    carried_text = (
        "\n".join(f"  - {c}" for c in carried)
        if carried else "  - none (this is a fresh request, not a follow-up)"
    )

    # User-pinned parameters ("if I fix X, recompute the rest") + any assumptions.
    user_fixed = payload.get("user_fixed", [])
    if user_fixed:
        fixed_text = "\n".join(
            f"  - {f['label']}: {f.get('display_value', f['value'])} {f['unit']} (FIXED BY OPERATOR)"
            for f in user_fixed
        )
    else:
        fixed_text = "  - none (operator did not pin any parameter)"

    warnings = payload.get("override_warnings", [])
    warnings_text = "\n".join(f"  - {w}" for w in warnings) or "  - none"

    cc = payload.get("consumable_change")
    if cc:
        consumable_change_text = (
            f"  - The operator changed wire diameter from {cc['table_diameter']} mm to "
            f"{cc['new_diameter']} mm.\n"
            f"  - Current / voltage / travel-speed windows DO NOT change with wire size — "
            f"they are set by the joint (material + thickness). Say this explicitly.\n"
            f"  - At the same {cc['optimized_wire_feed']} m/min wire feed, deposition is now "
            f"~{cc['deposition_at_optimized_wire_feed']} g/min "
            f"(vs ~{cc['deposition_standard_wire']} g/min with the {cc['table_diameter']} mm wire).\n"
            f"  - To hold the SAME deposit with the {cc['new_diameter']} mm wire, wire feed "
            f"should be ~{cc['wire_feed_for_same_deposition']} m/min."
        )
    else:
        consumable_change_text = "  - not applicable (wire diameter not changed)"

    defaulted = payload.get("defaulted", {})
    assumed = []
    if defaulted.get("material"):
        assumed.append(f"material assumed to be {payload.get('material_display', 'mild steel')} (not specified)")
    if defaulted.get("thickness"):
        assumed.append(f"thickness assumed to be {payload.get('thickness_mm', '?')} mm (not specified)")
    assumptions_text = "; ".join(assumed) if assumed else "none — material and thickness were specified"

    sens_lines = []
    for row in payload.get("sensitivity", []):
        name = row["parameter"]
        base = row["base_value"]
        up, down = row["up"], row["down"]
        sens_lines.append(
            f"  - {name} (currently {base}, step {row['step']}):\n"
            f"      raise to {up['value']}: heat_input -> {up['heat_input']} kJ/mm "
            f"({up['heat_input_change']}); deposition_rate -> "
            f"{'N/A' if up['deposition_rate'] is None else up['deposition_rate']}; "
            f"effects of raising {name}: {_effects_to_text(row['effects']['increase'])}; "
            f"heat-input-driven effects: {_effects_to_text(up['heat_input_effects'])}\n"
            f"      lower to {down['value']}: heat_input -> {down['heat_input']} kJ/mm "
            f"({down['heat_input_change']}); deposition_rate -> "
            f"{'N/A' if down['deposition_rate'] is None else down['deposition_rate']}; "
            f"effects of lowering {name}: {_effects_to_text(row['effects']['decrease'])}; "
            f"heat-input-driven effects: {_effects_to_text(down['heat_input_effects'])}"
        )
    sensitivity_text = "\n".join(sens_lines) or "  None available"

    return PARAM_ADVICE_PROMPT.format(
        question=payload.get("question", ""),
        process_display=payload.get("process_display", payload.get("process", "")),
        material_display=payload.get("material_display", ""),
        thickness_mm=payload.get("thickness_mm", "?"),
        band=payload.get("band", ""),
        carried_text=carried_text,
        fixed_text=fixed_text,
        assumptions_text=assumptions_text,
        warnings_text=warnings_text,
        ranges_text=ranges_text,
        optimized_text=optimized_text,
        metrics_text=metrics_text,
        efficiency_score=payload.get("efficiency_score", "?"),
        consumable_text=consumable_text,
        sensitivity_text=sensitivity_text,
        notes=payload.get("notes", ""),
    )


def _effects_to_text(effects: list[dict]) -> str:
    if not effects:
        return "none listed"
    return "; ".join(f"{e['effect']} {e['change']}" for e in effects)


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
