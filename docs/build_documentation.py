"""
Generate a professional PDF project-documentation report.

Run:
    .venv\\Scripts\\python.exe docs\\build_documentation.py

Output:
    docs/Welding_Chatbot_Project_Documentation.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ----------------------------------------------------------------------------
# Palette — industrial / welding theme (deep steel blue + arc amber accent)
# ----------------------------------------------------------------------------
NAVY = colors.HexColor("#1B2A41")     # deep steel blue
STEEL = colors.HexColor("#2E4A6B")    # mid blue
ACCENT = colors.HexColor("#E8A33D")   # arc amber
LIGHT = colors.HexColor("#F2F5F8")    # near-white panel
GREY = colors.HexColor("#5B6B7B")
RULE = colors.HexColor("#C9D3DD")
GREEN = colors.HexColor("#2E7D52")

OUT = Path(__file__).parent / "Welding_Chatbot_Project_Documentation.pdf"

# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------
ss = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                    fontSize=17, textColor=NAVY, spaceBefore=18, spaceAfter=8,
                    leading=21)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                    fontSize=12.5, textColor=STEEL, spaceBefore=12, spaceAfter=5,
                    leading=16)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=10, textColor=colors.HexColor("#222B36"),
                      leading=15, alignment=TA_JUSTIFY, spaceAfter=7)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=14, bulletIndent=4,
                        spaceAfter=3, alignment=TA_LEFT)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontName="Courier",
                      fontSize=8.5, textColor=colors.HexColor("#16324F"),
                      backColor=LIGHT, leading=12, leftIndent=8, rightIndent=8,
                      spaceBefore=4, spaceAfter=8, borderPadding=8)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=8.5, textColor=GREY,
                       alignment=TA_LEFT, spaceAfter=2)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=9, leading=12,
                      alignment=TA_LEFT, spaceAfter=0)
CELLH = ParagraphStyle("CELLH", parent=CELL, fontName="Helvetica-Bold",
                       textColor=colors.white, fontSize=9)
TOCITEM = ParagraphStyle("TOC", parent=BODY, fontSize=10.5, leading=20,
                         alignment=TA_LEFT, spaceAfter=0)


def P(t, style=BODY):
    return Paragraph(t, style)


def bullets(items, style=BULLET):
    return [Paragraph(f"• {i}", style) for i in items]


def kv_table(rows, col_widths):
    """Two-or-more-column table with header row styled in navy."""
    data = []
    for r in rows:
        data.append([Paragraph(str(c), CELLH if i_row == 0 else CELL)
                     for c in r] if False else r)
    # build with paragraph cells
    body = []
    for ri, r in enumerate(rows):
        line = []
        for c in r:
            if ri == 0:
                line.append(Paragraph(str(c), CELLH))
            else:
                line.append(Paragraph(str(c), CELL))
        body.append(line)
    t = Table(body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def code_block(text):
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace(" ", "&nbsp;").replace("\n", "<br/>"))
    return Paragraph(safe, CODE)


# ----------------------------------------------------------------------------
# Page furniture (header rule + footer page number)
# ----------------------------------------------------------------------------
DOC_TITLE = "Welding Process Optimization Chatbot with Explainable AI"


def _later_pages(canvas, doc):
    canvas.saveState()
    w, h = A4
    # header
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.6)
    canvas.line(2 * cm, h - 1.5 * cm, w - 2 * cm, h - 1.5 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, h - 1.4 * cm, DOC_TITLE)
    # footer
    canvas.line(2 * cm, 1.4 * cm, w - 2 * cm, 1.4 * cm)
    canvas.drawString(2 * cm, 1.0 * cm, "Project Documentation")
    canvas.drawRightString(w - 2 * cm, 1.0 * cm, "Page %d" % (doc.page - 1))
    canvas.restoreState()


def _cover_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # top navy band
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 6.2 * cm, w, 6.2 * cm, fill=1, stroke=0)
    # amber accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 6.45 * cm, w, 0.25 * cm, fill=1, stroke=0)
    # bottom band
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, w, 2.3 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 2.3 * cm, w, 0.12 * cm, fill=1, stroke=0)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(2 * cm, h - 2.1 * cm, "PROJECT DOCUMENTATION")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(ACCENT)
    canvas.drawString(2 * cm, h - 2.7 * cm,
                      "Smart Manufacturing  ·  Industry 4.0  ·  Explainable AI")

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 22)
    canvas.drawString(2 * cm, h - 4.1 * cm, "LLM-Enhanced, Multimodal")
    canvas.drawString(2 * cm, h - 4.95 * cm, "Welding Process Optimization")

    # footer band text
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, 1.45 * cm,
                      "Department of Mechatronics & Industrial Engineering (MIE), CUET")
    canvas.drawRightString(w - 2 * cm, 1.45 * cm, "Version 1.0  ·  June 2026")
    canvas.restoreState()


# ----------------------------------------------------------------------------
# Build the story
# ----------------------------------------------------------------------------
story = []

# ---- COVER (content under the painted bands) ----
story.append(Spacer(1, 7.3 * cm))

meta = [
    ["Project", "LLM-Enhanced, Multimodal Welding Process Optimization with Explainable AI"],
    ["Domain", "MIG / MAG / TIG / SMAW arc welding — smart manufacturing, Industry 4.0"],
    ["Type", "Operator-facing diagnostic & advisory chatbot (MVP, implemented & validated)"],
    ["Author", "Tasnimur Rahman Fayad  (ID: 2009026)"],
    ["Supervisor", "Kazi Naimur Rahman — Assistant Professor, Dept. of MIE, CUET"],
    ["Status", "MVP implemented and validated — 88/88 unit tests passing"],
    ["Version", "1.0  ·  June 2026"],
]
mt = Table([[Paragraph(f"<b>{k}</b>", CELL), Paragraph(v, CELL)] for k, v in meta],
           colWidths=[3.4 * cm, 12.6 * cm])
mt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, -1), LIGHT),
    ("GRID", (0, 0), (-1, -1), 0.5, RULE),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(mt)
story.append(Spacer(1, 0.6 * cm))
story.append(P(
    "<i>This document describes the project for academic review: what it is, the problem it "
    "solves, what was implemented, the technologies used, and the validation results.</i>",
    SMALL))
story.append(PageBreak())

# ---- TABLE OF CONTENTS ----
story.append(P("Table of Contents", H1))
toc = [
    "1.  Executive Summary",
    "2.  Problem Statement &amp; Motivation",
    "3.  Solution Overview",
    "4.  System Architecture",
    "5.  The Four Answer Routes",
    "6.  Machine Learning &amp; Anomaly Detection",
    "7.  Explainable AI (XAI) Layer",
    "8.  Retrieval-Augmented Generation (RAG) Knowledge Layer",
    "9.  LLM Synthesis Layer",
    "10. Technology Stack",
    "11. Dataset",
    "12. Evaluation &amp; Results",
    "13. User Interfaces",
    "14. How to Run",
    "15. Limitations &amp; Future Work",
    "16. Conclusion",
]
for item in toc:
    story.append(P(item, TOCITEM))
story.append(PageBreak())

# ---- 1. EXECUTIVE SUMMARY ----
story.append(P("1.  Executive Summary", H1))
story.append(P(
    "This project is an <b>operator-facing intelligent chatbot</b> for arc-welding shops that "
    "answers natural-language questions about welding operations. It unifies three heterogeneous "
    "data sources common to modern welding cells — <b>time-series sensor feeds</b>, structured "
    "<b>event/production logs</b>, and free-text <b>operator notes</b> — and combines classical "
    "machine learning, welding physics, explainable-AI techniques, and a Large Language Model (LLM) "
    "into a single conversational assistant.", BODY))
story.append(P(
    "The central design principle is <b>trustworthy, hallucination-resistant AI</b>. In a "
    "safety-critical welding context, fabricated numbers are unacceptable. Therefore every numeric "
    "output is computed by <b>deterministic physics or a trained ML model — never by the LLM</b>. "
    "Knowledge answers are grounded in a <b>retrieval-augmented (RAG) corpus with inline source "
    "citations</b>. The LLM is used <i>only</i> to phrase grounded results in clear, plain language. "
    "Every recommendation is accompanied by an explanation (SHAP feature attribution, parameter "
    "sensitivity tables, confidence scores, or cited sources).", BODY))
story.append(P(
    "The system was validated on a controlled synthetic, tri-modal dataset. It achieves an anomaly "
    "detection <b>F1 of 0.95</b> and <b>ROC-AUC of 0.999</b> under 5-fold cross-validation, "
    "<b>100% in-window compliance</b> for recommended weld parameters, <b>zero fabricated numbers</b>, "
    "and passes <b>88 of 88</b> unit tests.", BODY))

# ---- 2. PROBLEM STATEMENT ----
story.append(P("2.  Problem Statement &amp; Motivation", H1))
story.append(P(
    "Modern welding shops generate large volumes of heterogeneous data, yet operators still struggle to:", BODY))
story.extend(bullets([
    "choose <b>optimal weld parameters</b> (current, voltage, travel speed, wire feed) for a given job;",
    "<b>diagnose faults</b> quickly when a station behaves abnormally;",
    "obtain <b>trustworthy, explainable</b> guidance they can act on with confidence.",
]))
story.append(P(
    "Conventional machine-learning tools behave as opaque <b>black boxes</b> that output a prediction "
    "without justification, while raw LLMs can <b>hallucinate unsafe numbers</b>. Neither is acceptable "
    "on a shop floor where an incorrect heat-input figure can ruin a weld or create a safety hazard. "
    "This project addresses that gap by separating <b>computation</b> (physics + ML, fully auditable) "
    "from <b>communication</b> (the LLM, phrasing only), and by attaching an explanation to every answer.", BODY))

# ---- 3. SOLUTION OVERVIEW ----
story.append(P("3.  Solution Overview", H1))
story.append(P(
    "The chatbot answers welding and manufacturing questions through a set of specialised, grounded "
    "routes. An <b>intent guard</b> first rejects out-of-scope questions (general knowledge, jokes, "
    "geography, etc.). In-scope questions are then dispatched by a <b>router</b> to the appropriate "
    "engine:", BODY))
story.append(kv_table([
    ["Capability", "How it is grounded", "Explainability"],
    ["Parameter optimization", "Welding physics + deterministic grid search", "Sensitivity table"],
    ["Fault diagnosis", "Supervised ML on fused sensor/log/note data", "SHAP + LIME"],
    ["Welding knowledge", "Hybrid RAG retrieval over a welding corpus", "Inline [S#] citations"],
    ["General manufacturing", "LLM advisory (numbers flagged to verify)", "Stated assumptions"],
], [4.2 * cm, 7.2 * cm, 4.6 * cm]))
story.append(Spacer(1, 4))
story.append(P(
    "The result is an assistant that is <b>transparent, traceable, grounded, and "
    "hallucination-resistant</b>: numeric outputs come from physics/ML, knowledge answers from cited "
    "retrieval, and the LLM never invents figures.", BODY))

# ---- 4. SYSTEM ARCHITECTURE ----
story.append(P("4.  System Architecture", H1))
story.append(P(
    "The system follows a clean pipeline: <b>intent guard → router → route-specific engine → LLM "
    "synthesis → answer</b>. Numeric routes are fully deterministic and auditable; only the final "
    "phrasing passes through the LLM.", BODY))
story.append(code_block(
    "User question\n"
    "     |\n"
    "     v\n"
    "[Intent guard]  -------- out-of-scope --------> polite refusal (no engine runs)\n"
    "     | in-scope\n"
    "     v\n"
    "[Router]\n"
    "     |------------------+------------------+---------------------+\n"
    "     v                  v                  v                     v\n"
    "  PARAM             ANOMALY           KNOWLEDGE (RAG)        GENERAL\n"
    "  physics +         sensor+log+note   corpus: KB + docs      manufacturing\n"
    "  grid search       -> RandomForest   -> embed + hybrid      LLM advisor\n"
    "  -> HI, DR         -> SHAP / LIME       retrieval (sem+kw)\n"
    "  -> sensitivity    -> root cause +    -> cited passages [S#]\n"
    "                       confidence\n"
    "     |------------------+------------------+---------------------+\n"
    "                        |\n"
    "                        v\n"
    "          [LLM synthesis]  (Claude / Groq / Ollama) -- phrasing only\n"
    "                        |\n"
    "                        v\n"
    "   Plain-language answer + explainability  (Streamlit UI / REST API)"))
story.append(P("4.1  Key Modules", H2))
story.append(kv_table([
    ["Module", "Location", "Responsibility"],
    ["Intent classifier", "src/chat/intent.py", "Scope gate (keyword + regex, no LLM call)"],
    ["Router", "src/api/pipeline.py", "route_question() — picks the engine"],
    ["Parameter advisor", "src/reasoning/param_advisor.py", "Physics + grid-search optimization"],
    ["Anomaly detector", "src/model/anomaly.py", "RandomForest classifier"],
    ["DistilBERT detector", "src/model/bert_detector.py", "Fine-tuned transformer (optional 2nd detector)"],
    ["SHAP explainer", "src/explain/shap_explainer.py", "Feature attribution"],
    ["LIME / attention", "src/explain/", "Independent XAI cross-checks"],
    ["RAG retriever", "src/rag/ (corpus, index, retriever)", "Corpus build, embeddings, hybrid retrieval"],
    ["LLM synthesis", "src/chat/synthesize.py, prompts.py", "Narration, pluggable backends"],
    ["Frontend", "app/streamlit_app.py", "Chat UI (dark industrial theme)"],
    ["API", "src/api/main.py", "REST endpoints (FastAPI)"],
], [3.6 * cm, 5.2 * cm, 7.2 * cm]))

# ---- 5. THE FOUR ANSWER ROUTES ----
story.append(P("5.  The Four Answer Routes", H1))

story.append(P("5.1  Parameter Optimization (primary deliverable)", H2))
story.append(P(
    "Given a material, thickness and process, the advisor looks up the published standards window "
    "(AWS D1.1 / ISO 15614-1 style), then runs a <b>deterministic grid search</b> to find the "
    "best-efficiency settings. Two welding-physics quantities drive the objective:", BODY))
story.append(code_block(
    "Heat input:       HI = (eta * 60 * I * V) / (1000 * S)     [kJ/mm]\n"
    "Deposition rate:  DR = WFS * A * rho * eta_d               [g/min]\n\n"
    "Objective (maximised):  deposition_rate  x  heat_input_window_score\n"
    "where I = current, V = voltage, S = travel speed, WFS = wire feed speed."))
story.append(P(
    "It returns the optimal current/voltage/speed/wire-feed/gas-flow, the resulting heat input and "
    "deposition rate, an efficiency score, and a <b>sensitivity table</b> (each parameter is nudged "
    "&plusmn;1 step and the qualitative effect listed). For unsupported materials (e.g. titanium, copper) "
    "or invalid combinations (e.g. SMAW on aluminium) it returns an <b>honest error rather than "
    "fabricated numbers</b>.", BODY))
story.append(P(
    "<b>Example —</b> 5&nbsp;mm stainless steel, MIG: 187&nbsp;A, 24.2&nbsp;V, 450&nbsp;mm/min, "
    "9.3&nbsp;m/min wire feed; heat input 0.493&nbsp;kJ/mm (window 0.30–0.65); deposition "
    "78.5&nbsp;g/min; efficiency 8/10.", BODY))

story.append(P("5.2  Fault Diagnosis", H2))
story.append(P(
    "When a question references a specific station/line/time, the system loads and fuses the sensor, "
    "log and operator-note data for that station, detects whether the window is anomalous, classifies "
    "the fault, explains it with SHAP, maps it to a likely root cause and recommended action, and "
    "attaches a confidence score. Five fault types are supported: <b>arc instability, wire-feed fault, "
    "gas-flow failure, overheating, and underheat</b>.", BODY))

story.append(P("5.3  Welding Knowledge (RAG)", H2))
story.append(P(
    "Defect, troubleshooting, quality, productivity and cost questions are answered by a "
    "<b>hybrid retrieval-augmented generation</b> layer: the most relevant passages are retrieved from "
    "a welding corpus (curated knowledge base + drop-in reference documents) using combined "
    "semantic-embedding and keyword search, and the LLM answer is grounded strictly in those passages "
    "with inline <b>[S#] source citations</b>. (Detailed in Section 8.)", BODY))

story.append(P("5.4  General Manufacturing &amp; Simulation Integration", H2))
story.append(P(
    "Questions about non-welding manufacturing domains (CNC machining, injection moulding, casting, "
    "forming, additive manufacturing, quality/operations) are routed to a more capable LLM advisor "
    "(Claude Opus). Because there is no deterministic grounding here, the prompt instructs the model to "
    "flag specific numbers as <b>starting points to verify</b>. A further specialised persona answers "
    "<b>robotics / simulation-integration</b> questions (NVIDIA Isaac Sim, digital twins, sensors, "
    "teleoperation, data acquisition) relevant to the project's future direction.", BODY))

# ---- 6. ML ----
story.append(P("6.  Machine Learning &amp; Anomaly Detection", H1))
story.append(P(
    "The diagnosis engine runs a <b>load → preprocess → fuse → detect → explain → reason → confidence</b> "
    "pipeline. Per-modality preprocessing imputes and denoises sensor signals into 30-minute windows, "
    "counts log event types and alarm flags, and embeds operator notes. The modalities are then "
    "<b>temporally aligned and fused</b> (pandas merge_asof) into a single feature matrix.", BODY))
story.append(P(
    "The primary detector is a <b>RandomForest classifier</b> (scikit-learn) — chosen because it is "
    "accurate on tabular fused features and natively compatible with the SHAP TreeExplainer. An "
    "<b>optional fine-tuned DistilBERT</b> transformer detector is also implemented: each fused window "
    "is serialised to a short text description and classified by a fine-tuned sequence model, providing "
    "a like-for-like comparison and demonstrating the LLM fine-tuning path. A <b>confidence score</b> "
    "combines model certainty and explanation agreement:", BODY))
story.append(code_block("confidence = 0.7 x anomaly_probability  +  0.3 x SHAP_agreement"))

# ---- 7. XAI ----
story.append(P("7.  Explainable AI (XAI) Layer", H1))
story.append(P(
    "Explainability is a first-class requirement — no prediction is presented without justification. "
    "Three complementary techniques are used:", BODY))
story.extend(bullets([
    "<b>SHAP (TreeExplainer)</b> — primary attribution; surfaces the top sensor channels that drove "
    "each anomaly prediction, with a waterfall plot in the UI.",
    "<b>LIME</b> — an independent local-surrogate cross-check, run as a second opinion on the same window.",
    "<b>Attention visualisation</b> — highlights the salient tokens in the relevant operator note.",
]))
story.append(P(
    "A heuristic <b>root-cause mapper</b> then matches the SHAP driver pattern (plus log event codes) to "
    "a ranked list of likely causes, and a <b>recommendation engine</b> maps each fault type to a "
    "primary corrective action and urgency level (config/cause_action_map.yaml).", BODY))

# ---- 8. RAG ----
story.append(P("8.  Retrieval-Augmented Generation (RAG) Knowledge Layer", H1))
story.append(P(
    "The welding-knowledge route upgrades brittle keyword matching to <b>semantic retrieval with grounded, "
    "cited generation</b>. The pipeline has three stages:", BODY))
story.extend(bullets([
    "<b>Corpus</b> (src/rag/corpus.py) — one passage per topic from the curated knowledge YAML, plus "
    "reference documents in data/knowledge_docs/ chunked by section.",
    "<b>Index</b> (src/rag/index.py) — a local sentence-transformers encoder (all-MiniLM-L6-v2) produces "
    "L2-normalised embeddings persisted in models/rag_index/; cosine search via dot product. The index "
    "rebuilds automatically only when the corpus content hash changes.",
    "<b>Retriever</b> (src/rag/retriever.py) — hybrid ranking blends semantic and lexical signals and "
    "tags each returned passage for citation:",
]))
story.append(code_block("score = 0.7 x semantic(cosine)  +  0.3 x lexical(word overlap)\n"
                        "  -> top-k passages, each tagged [S1], [S2], ... for inline citation"))
story.append(P(
    "Key design choices: <b>local embeddings</b> (free, fast, offline-capable — no data leaves the "
    "machine, no cloud vector DB); <b>hybrid retrieval</b> (semantic catches paraphrases, lexical catches "
    "exact defect/parameter names); <b>inline [S#] citations</b> for full traceability; and a "
    "<b>keyword fallback</b> so the route degrades gracefully if the embedding model cannot load.", BODY))

# ---- 9. LLM ----
story.append(P("9.  LLM Synthesis Layer", H1))
story.append(P(
    "The LLM converts the structured engine output into a plain-language answer — and nothing more. "
    "A strict system prompt forbids it from producing any number not present in the computed payload. "
    "If no LLM backend is configured, a <b>deterministic text fallback</b> still returns the grounded "
    "result (computed values or cited RAG passages). The layer is backend-pluggable:", BODY))
story.append(kv_table([
    ["Backend", "Model (default)", "Role"],
    ["Anthropic Claude", "claude-haiku-4-5 (narration)", "Default — fast, low-cost phrasing"],
    ["Anthropic Claude", "claude-opus-4-8 (general route)", "Higher-capability advisory answers"],
    ["Groq", "llama-3.3-70b-versatile", "Fast hosted alternative"],
    ["Ollama", "llama3.2 (local)", "Fully offline operation"],
], [3.8 * cm, 6.0 * cm, 6.2 * cm]))
story.append(Spacer(1, 4))
story.append(P(
    "The backend is selected via the <font face='Courier'>SYNTHESIZER_BACKEND</font> environment "
    "variable, and falls back through the chain if the chosen backend has no credentials, so a demo "
    "always runs.", BODY))

# ---- 10. TECH STACK ----
story.append(P("10.  Technology Stack", H1))
story.append(kv_table([
    ["Area", "Technologies"],
    ["Language / runtime", "Python 3.11 (pure-Python, Windows-friendly, no GPU required)"],
    ["Data", "NumPy, pandas, PyArrow (Parquet)"],
    ["Machine learning", "scikit-learn (RandomForest), joblib"],
    ["NLP / transformers", "sentence-transformers (all-MiniLM-L6-v2), Hugging Face Transformers, "
                           "Accelerate (DistilBERT fine-tuning)"],
    ["Explainability (XAI)", "SHAP, LIME, Matplotlib"],
    ["LLM backends", "Anthropic (Claude), Groq (Llama 3.3), Ollama (local), python-dotenv"],
    ["Validation / config", "Pydantic, PyYAML"],
    ["Backend API", "FastAPI, Uvicorn"],
    ["Frontend", "Streamlit (dark industrial theme)"],
    ["Testing", "pytest, pytest-cov (88 tests)"],
], [4.0 * cm, 12.0 * cm]))

# ---- 11. DATASET ----
story.append(P("11.  Dataset", H1))
story.append(P(
    "A fully <b>synthetic, tri-modal dataset</b> is generated programmatically with NumPy and pandas. "
    "Synthetic generation was a deliberate decision: real factory data is sensitive and rarely tri-modal, "
    "whereas synthetic generation gives full control over modalities, anomaly types and "
    "<b>exact ground-truth labels</b>, making evaluation meaningful. The pipeline is data-agnostic — real "
    "CSVs can be swapped in directly.", BODY))
story.append(kv_table([
    ["Modality / file", "Contents", "Size"],
    ["sensors.csv", "5 channels: current, voltage, speed, wire-feed, gas-flow", "30,240 rows"],
    ["logs.csv", "Structured event / maintenance / alarm records", "278 rows"],
    ["notes.csv", "Templated free-text operator shift notes", "31 rows"],
    ["ground_truth.csv", "Per-window labels: anomaly type + root cause", "30 windows"],
    ["fused.parquet", "Aligned & fused 30-min feature windows", "2,010 windows (62 anomalous)"],
], [4.0 * cm, 8.0 * cm, 4.0 * cm]))
story.append(Spacer(1, 4))
story.append(P(
    "Scope: 3 welding stations &times; 7 days at 1-minute resolution; materials = mild steel / stainless "
    "steel / aluminium; processes = MIG/MAG (GMAW), TIG (GTAW), SMAW (stick); five injected fault types.", BODY))

# ---- 12. EVALUATION ----
story.append(P("12.  Evaluation &amp; Results", H1))
story.append(P(
    "All metrics are computed on the synthetic dataset under 5-fold stratified cross-validation "
    "(labels known exactly). Headline results:", BODY))
story.append(kv_table([
    ["Metric", "Target", "Result"],
    ["Anomaly detection F1 (5-fold CV)", "&ge; 0.85", "0.95"],
    ["Precision / Recall", "—", "0.92 / 0.98"],
    ["ROC-AUC", "&ge; 0.90", "0.999"],
    ["SHAP driver alignment (overall)", "high", "82% (100% for 4 of 5 fault types)"],
    ["Root-cause ranking (MRR / Top-3)", "—", "0.878 / 100%"],
    ["Parameter advisor in-window compliance", "100%", "100% (72 setpoints)"],
    ["Confidence calibration (high band)", "—", "98% accurate"],
    ["Fabricated numbers in answers", "0", "0"],
    ["Unit tests passing", "all", "88 / 88"],
], [8.4 * cm, 3.0 * cm, 4.6 * cm]))
story.append(Spacer(1, 4))
story.append(P(
    "The optional fine-tuned DistilBERT detector matches the RandomForest baseline on the same "
    "train/test split, validating the transformer fine-tuning path. The near-perfect ROC-AUC reflects "
    "the cleanly separable synthetic faults and would be lower on noisier real-world data.", SMALL))

# ---- 13. INTERFACES ----
story.append(P("13.  User Interfaces", H1))
story.extend(bullets([
    "<b>Streamlit chat UI</b> — a dark industrial-themed conversational interface with parameter cards, "
    "SHAP charts, confidence badges, knowledge cards and retrieved-source panels.",
    "<b>FastAPI REST backend</b> — programmatic access via <font face='Courier'>POST /chat</font>, "
    "<font face='Courier'>POST /pipeline/raw</font>, and <font face='Courier'>GET /health</font>.",
]))

# ---- 14. HOW TO RUN ----
story.append(P("14.  How to Run", H1))
story.append(code_block(
    "# 1. install dependencies (Python 3.11 venv)\n"
    ".venv\\Scripts\\python.exe -m pip install -r requirements.txt\n\n"
    "# 2. configure a backend (copy and edit the env file)\n"
    "copy .env.example .env        # set SYNTHESIZER_BACKEND and an API key\n\n"
    "# 3. generate the synthetic dataset\n"
    ".venv\\Scripts\\python.exe src\\data\\generate_synthetic.py\n\n"
    "# 4. run the chatbot UI            ->  http://localhost:8501\n"
    ".venv\\Scripts\\python.exe -m streamlit run app\\streamlit_app.py\n\n"
    "# 5. (optional) run the REST API   ->  http://localhost:8000\n"
    ".venv\\Scripts\\python.exe -m uvicorn src.api.main:app --reload\n\n"
    "# 6. run the test suite            ->  88 passing\n"
    ".venv\\Scripts\\python.exe -m pytest tests\\ -q"))

# ---- 15. LIMITATIONS ----
story.append(P("15.  Limitations &amp; Future Work", H1))
story.append(P("<b>Current limitations</b>", H2))
story.extend(bullets([
    "Validated on synthetic (not real factory) data; near-perfect AUC is optimistic for production noise.",
    "The LLM is narration-only (not fine-tuned for domain reasoning) — deliberate, to prevent hallucinated numbers.",
    "Single-fault-per-window detection; no visual/image weld-defect inspection.",
    "No physical hardware / PLC / SCADA integration or real-time closed-loop control.",
]))
story.append(P("<b>Planned future work</b>", H2))
story.extend(bullets([
    "Validate on real welding-cell data and broaden materials/processes (titanium, FCAW, SAW).",
    "Vision-language model for weld-defect images; multi-label fault detection.",
    "Live MES / sensor-stream integration with a human-in-the-loop confirmation UI.",
    "Robotics / digital-twin integration (NVIDIA Isaac Sim) for sim-to-real welding-cell automation.",
]))

# ---- 16. CONCLUSION ----
story.append(P("16.  Conclusion", H1))
story.append(P(
    "This project delivers a working MVP of an explainable, multimodal welding assistant that meets its "
    "core thesis goal: <b>transparent, traceable, grounded, hallucination-resistant</b> guidance. By "
    "strictly separating computation (welding physics and machine learning) from communication (the LLM), "
    "and by grounding every knowledge answer in cited retrieval, the system provides shop-floor advice "
    "that operators can trust — each recommendation backed by an explanation, a confidence score, or a "
    "source citation. The architecture is modular and data-agnostic, providing a solid foundation for the "
    "future work outlined above.", BODY))

# ----------------------------------------------------------------------------
# Document assembly with two page templates (cover + content)
# ----------------------------------------------------------------------------
doc = BaseDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=2 * cm, bottomMargin=2 * cm,
    title="Welding Process Optimization Chatbot — Project Documentation",
    author="Tasnimur Rahman Fayad",
)
frame_cover = Frame(0, 0, A4[0], A4[1], id="cover",
                    leftPadding=2 * cm, rightPadding=2 * cm,
                    topPadding=2 * cm, bottomPadding=2 * cm)
frame_body = Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 4 * cm, id="body")
doc.addPageTemplates([
    PageTemplate(id="Cover", frames=[frame_cover], onPage=_cover_page),
    PageTemplate(id="Content", frames=[frame_body], onPage=_later_pages),
])

# switch to the Content template after the cover
from reportlab.platypus import NextPageTemplate  # noqa: E402

story.insert(1, NextPageTemplate("Content"))

doc.build(story)
print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
