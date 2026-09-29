"""
Intent Classification Stress Benchmark Dataset (Step 8).

Defines 100 hand-curated, domain-grounded test queries across the 5 unified
intent categories recognized by the existing system:
  1. out_of_scope (unsupported / non-manufacturing chitchat, trivia, general QA)
  2. anomaly      (station/machine-specific telemetry, alerts, and fault diagnosis)
  3. param        (material + thickness welding parameter optimization)
  4. knowledge    (welding defect remedies, metallurgy, economics, robotics sim)
  5. general      (non-welding manufacturing: CNC, molding, casting, 3D print, lean)

Covering 12 realistic operator-language perturbation types:
  - clean_direct
  - short_query
  - paraphrase
  - informal_slang
  - typos_spelling
  - abbreviations
  - code_switching (mixed Bangla-English common on regional shop floors)
  - word_order
  - noisy_redundant
  - multi_keyword
  - short_ambiguous
  - unsupported_out_of_scope
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).resolve().parent.parent

INTENT_TAXONOMY: List[str] = [
    "out_of_scope",
    "anomaly",
    "param",
    "knowledge",
    "general",
]

PERTURBATION_TYPES: List[str] = [
    "clean_direct",
    "short_query",
    "paraphrase",
    "informal_slang",
    "typos_spelling",
    "abbreviations",
    "code_switching",
    "word_order",
    "noisy_redundant",
    "multi_keyword",
    "short_ambiguous",
    "unsupported_out_of_scope",
]

DIFFICULTY_LEVELS: List[str] = [
    "easy",
    "medium",
    "hard",
]

# 100 Hand-Curated Stress Queries (20 per intent category)
BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    # =========================================================================
    # INTENT: out_of_scope (20 queries)
    # =========================================================================
    {
        "query_id": "OOS_01",
        "text": "Who won the FIFA World Cup in 2022?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "General sports trivia with no manufacturing keywords or patterns."
    },
    {
        "query_id": "OOS_02",
        "text": "Tell me a funny joke about office work.",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "General humor / chitchat request."
    },
    {
        "query_id": "OOS_03",
        "text": "What is the capital city of Bangladesh?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Geography trivia with no industrial context."
    },
    {
        "query_id": "OOS_04",
        "text": "Calculate 254 multiplied by 18.",
        "expected_intent": "out_of_scope",
        "perturbation_type": "short_query",
        "difficulty": "easy",
        "rationale": "Pure arithmetic calculation request."
    },
    {
        "query_id": "OOS_05",
        "text": "Write a short poem about the rainy season.",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Creative writing prompt."
    },
    {
        "query_id": "OOS_06",
        "text": "What is the weather forecast for London this weekend?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Meteorology / weather inquiry."
    },
    {
        "query_id": "OOS_07",
        "text": "Can you recommend a good Italian restaurant nearby?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Dining recommendation request."
    },
    {
        "query_id": "OOS_08",
        "text": "Translate this email from English to German.",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Linguistic translation request."
    },
    {
        "query_id": "OOS_09",
        "text": "How do I bake a gluten-free sourdough bread at home?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Culinary recipe inquiry."
    },
    {
        "query_id": "OOS_10",
        "text": "What movies are premiering in theaters tonight?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Entertainment inquiry."
    },
    {
        "query_id": "OOS_11",
        "text": "ajk brishti hobe naki re bhai?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "code_switching",
        "difficulty": "medium",
        "rationale": "Phonetic Bangla asking about weather with no English industrial terms."
    },
    {
        "query_id": "OOS_12",
        "text": "Can you write a Python function to sort an array using quicksort?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "medium",
        "rationale": "Computer science software coding prompt, unrelated to factory operations."
    },
    {
        "query_id": "OOS_13",
        "text": "Hello assistant how are you feeling this bright morning?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "noisy_redundant",
        "difficulty": "easy",
        "rationale": "Conversational greeting."
    },
    {
        "query_id": "OOS_14",
        "text": "h r u doin 2day",
        "expected_intent": "out_of_scope",
        "perturbation_type": "informal_slang",
        "difficulty": "medium",
        "rationale": "SMS shorthand greeting."
    },
    {
        "query_id": "OOS_15",
        "text": "Who is currently the prime minister of Japan?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Political trivia."
    },
    {
        "query_id": "OOS_16",
        "text": "My dog has a fever and won't eat what should I do?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Veterinary advice request."
    },
    {
        "query_id": "OOS_17",
        "text": "Explain the philosophical difference between nihilism and stoicism.",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "easy",
        "rationale": "Philosophy inquiry."
    },
    {
        "query_id": "OOS_18",
        "text": "best smartphone under 500 dollars",
        "expected_intent": "out_of_scope",
        "perturbation_type": "short_ambiguous",
        "difficulty": "hard",
        "rationale": "Contains 'best' but refers to consumer electronics, not welding settings."
    },
    {
        "query_id": "OOS_19",
        "text": "What is the fastest car in the world right now?",
        "expected_intent": "out_of_scope",
        "perturbation_type": "unsupported_out_of_scope",
        "difficulty": "medium",
        "rationale": "Automotive trivia with no factory or manufacturing context."
    },
    {
        "query_id": "OOS_20",
        "text": "wether forcast 2morow mornin pls",
        "expected_intent": "out_of_scope",
        "perturbation_type": "typos_spelling",
        "difficulty": "easy",
        "rationale": "Misspelled weather request."
    },

    # =========================================================================
    # INTENT: anomaly (20 queries)
    # =========================================================================
    {
        "query_id": "ANOM_01",
        "text": "Why did station_1 stop welding at 14:20?",
        "expected_intent": "anomaly",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Standard anomaly diagnosis naming station_1 and timestamp."
    },
    {
        "query_id": "ANOM_02",
        "text": "What caused the vibration alarm on station 2?",
        "expected_intent": "anomaly",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Station 2 anomaly diagnosis with sensor vibration alarm."
    },
    {
        "query_id": "ANOM_03",
        "text": "Is machine_3 operating normally or is there an alert?",
        "expected_intent": "anomaly",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Telemetry status inquiry for machine_3."
    },
    {
        "query_id": "ANOM_04",
        "text": "line 2 alarm",
        "expected_intent": "anomaly",
        "perturbation_type": "short_query",
        "difficulty": "easy",
        "rationale": "Telegraphic query naming line 2 and alarm."
    },
    {
        "query_id": "ANOM_05",
        "text": "s1 overheat",
        "expected_intent": "anomaly",
        "perturbation_type": "short_query",
        "difficulty": "medium",
        "rationale": "Abbreviated station 's1' with overheating fault keyword."
    },
    {
        "query_id": "ANOM_06",
        "text": "Please check whether station 2 has registered any unusual voltage fluctuations during the last hour.",
        "expected_intent": "anomaly",
        "perturbation_type": "paraphrase",
        "difficulty": "medium",
        "rationale": "Polite paraphrase asking for station 2 telemetry diagnostics."
    },
    {
        "query_id": "ANOM_07",
        "text": "station 1 is acting crazy and running super hot right now",
        "expected_intent": "anomaly",
        "perturbation_type": "informal_slang",
        "difficulty": "medium",
        "rationale": "Colloquial operator description of station 1 thermal anomaly."
    },
    {
        "query_id": "ANOM_08",
        "text": "sttion_1 triggred an alrm at 10:15",
        "expected_intent": "anomaly",
        "perturbation_type": "typos_spelling",
        "difficulty": "hard",
        "rationale": "Typos in station and alarm words, stress-testing regex robustness."
    },
    {
        "query_id": "ANOM_09",
        "text": "machine 2 vib alert at 15:30",
        "expected_intent": "anomaly",
        "perturbation_type": "abbreviations",
        "difficulty": "medium",
        "rationale": "Abbreviated vibration ('vib') on machine 2."
    },
    {
        "query_id": "ANOM_10",
        "text": "station 3 e hot alarm bajtese keno?",
        "expected_intent": "anomaly",
        "perturbation_type": "code_switching",
        "difficulty": "medium",
        "rationale": "Bangla-English code-mix asking why alarm is ringing on station 3."
    },
    {
        "query_id": "ANOM_11",
        "text": "At 09:45 why did line 1 suddenly shut down and trip the breaker?",
        "expected_intent": "anomaly",
        "perturbation_type": "word_order",
        "difficulty": "medium",
        "rationale": "Time-first word order diagnosing line 1 trip."
    },
    {
        "query_id": "ANOM_12",
        "text": "Hey supervisor told me to ask you if station_2 had an arc loss anomaly around 11:30 earlier today.",
        "expected_intent": "anomaly",
        "perturbation_type": "noisy_redundant",
        "difficulty": "medium",
        "rationale": "Conversational padding wrapping station_2 fault inquiry."
    },
    {
        "query_id": "ANOM_13",
        "text": "station 1 has high spatter and voltage variance what is the fault?",
        "expected_intent": "anomaly",
        "perturbation_type": "multi_keyword",
        "difficulty": "hard",
        "rationale": "Combines station 1 identifier with defect keywords (spatter, voltage)."
    },
    {
        "query_id": "ANOM_14",
        "text": "status station 3",
        "expected_intent": "anomaly",
        "perturbation_type": "short_ambiguous",
        "difficulty": "medium",
        "rationale": "Minimalist status check on station 3."
    },
    {
        "query_id": "ANOM_15",
        "text": "Why did the motor on machine 1 trip during the morning run?",
        "expected_intent": "anomaly",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Direct machine 1 motor trip diagnostic query."
    },
    {
        "query_id": "ANOM_16",
        "text": "m2 running hot",
        "expected_intent": "anomaly",
        "perturbation_type": "short_query",
        "difficulty": "medium",
        "rationale": "Abbreviated machine 2 ('m2') thermal issue."
    },
    {
        "query_id": "ANOM_17",
        "text": "diagnose fault on line_3",
        "expected_intent": "anomaly",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Clear directive targeting line_3 fault diagnosis."
    },
    {
        "query_id": "ANOM_18",
        "text": "station_2 temperature spiked past 85 degrees",
        "expected_intent": "anomaly",
        "perturbation_type": "paraphrase",
        "difficulty": "medium",
        "rationale": "Sensor reading report on station_2."
    },
    {
        "query_id": "ANOM_19",
        "text": "line 1 e wire feed theme gese keno?",
        "expected_intent": "anomaly",
        "perturbation_type": "code_switching",
        "difficulty": "hard",
        "rationale": "Code-mixed Bangla asking why wire feed stopped on line 1."
    },
    {
        "query_id": "ANOM_20",
        "text": "Is there any anomaly recorded for station 1 on the 14:00 shift?",
        "expected_intent": "anomaly",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Standard shift anomaly verification on station 1."
    },

    # =========================================================================
    # INTENT: param (20 queries)
    # =========================================================================
    {
        "query_id": "PARAM_01",
        "text": "Recommend optimal welding parameters for 6 mm mild steel MIG.",
        "expected_intent": "param",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Direct parameter optimization query with material (mild steel) and thickness (6 mm)."
    },
    {
        "query_id": "PARAM_02",
        "text": "What are the best settings for 4mm aluminum TIG welding?",
        "expected_intent": "param",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Contains material (aluminum) and thickness (4mm)."
    },
    {
        "query_id": "PARAM_03",
        "text": "What current and voltage should I use for 5 mm stainless steel plate?",
        "expected_intent": "param",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Material (stainless steel) and thickness (5 mm) parameter advisory."
    },
    {
        "query_id": "PARAM_04",
        "text": "5mm steel settings",
        "expected_intent": "param",
        "perturbation_type": "short_query",
        "difficulty": "easy",
        "rationale": "Telegraphic query with thickness (5mm) and material (steel)."
    },
    {
        "query_id": "PARAM_05",
        "text": "3 mm stainless mig params",
        "expected_intent": "param",
        "perturbation_type": "abbreviations",
        "difficulty": "medium",
        "rationale": "Abbreviated 'params' with 3 mm and stainless."
    },
    {
        "query_id": "PARAM_06",
        "text": "Could you provide recommended operating setpoints for welding 8 mm thick carbon steel plates?",
        "expected_intent": "param",
        "perturbation_type": "paraphrase",
        "difficulty": "medium",
        "rationale": "Polite paraphrase specifying 8 mm carbon steel."
    },
    {
        "query_id": "PARAM_07",
        "text": "Give me the fastest travel speed and juice for 10mm mild steel joint.",
        "expected_intent": "param",
        "perturbation_type": "informal_slang",
        "difficulty": "medium",
        "rationale": "Slang ('juice' for current/power) with 10mm mild steel."
    },
    {
        "query_id": "PARAM_08",
        "text": "optmize mig weld for 6mm mld stel",
        "expected_intent": "param",
        "perturbation_type": "typos_spelling",
        "difficulty": "hard",
        "rationale": "Multiple typos in optimize, mild, steel while keeping 6mm."
    },
    {
        "query_id": "PARAM_09",
        "text": "4mm ms plate parameters",
        "expected_intent": "param",
        "perturbation_type": "abbreviations",
        "difficulty": "medium",
        "rationale": "Abbreviation 'ms' for mild steel alongside 4mm."
    },
    {
        "query_id": "PARAM_10",
        "text": "5mm stainless sheet e best current voltage koto lagbe?",
        "expected_intent": "param",
        "perturbation_type": "code_switching",
        "difficulty": "medium",
        "rationale": "Bangla-English code-switching for 5mm stainless parameter setpoints."
    },
    {
        "query_id": "PARAM_11",
        "text": "For mild steel of 12 mm thickness what welding speed and wire feed do you recommend?",
        "expected_intent": "param",
        "perturbation_type": "word_order",
        "difficulty": "medium",
        "rationale": "Inverted word order with 12 mm and mild steel."
    },
    {
        "query_id": "PARAM_12",
        "text": "I am setting up a production run today and I need the exact parameter configuration for 2 mm mild steel sheet welding.",
        "expected_intent": "param",
        "perturbation_type": "noisy_redundant",
        "difficulty": "medium",
        "rationale": "Lengthy conversational setup containing 2 mm mild steel."
    },
    {
        "query_id": "PARAM_13",
        "text": "Recommend settings for 6 mm mild steel to prevent porosity and undercut.",
        "expected_intent": "param",
        "perturbation_type": "multi_keyword",
        "difficulty": "hard",
        "rationale": "Combines parameter optimization targets with defect keywords."
    },
    {
        "query_id": "PARAM_14",
        "text": "8mm aluminum",
        "expected_intent": "param",
        "perturbation_type": "short_ambiguous",
        "difficulty": "medium",
        "rationale": "Ultra-short material and thickness pair."
    },
    {
        "query_id": "PARAM_15",
        "text": "Suggest welding current and arc voltage for 15 mm mild steel joint.",
        "expected_intent": "param",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Standard parameter request for 15 mm mild steel."
    },
    {
        "query_id": "PARAM_16",
        "text": "best speed for 2 mm stainless steel",
        "expected_intent": "param",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Speed optimization inquiry for 2 mm stainless."
    },
    {
        "query_id": "PARAM_17",
        "text": "What wire feed speed should be set for 10 mm aluminium plate?",
        "expected_intent": "param",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Wire feed parameter query for 10 mm aluminium."
    },
    {
        "query_id": "PARAM_18",
        "text": "1.5 mm ss sheet tig parameter",
        "expected_intent": "param",
        "perturbation_type": "abbreviations",
        "difficulty": "medium",
        "rationale": "Abbreviations 'ss' (stainless) and 'tig' with 1.5 mm thickness."
    },
    {
        "query_id": "PARAM_19",
        "text": "3mm steel er jonno voltage current koto?",
        "expected_intent": "param",
        "perturbation_type": "code_switching",
        "difficulty": "medium",
        "rationale": "Bangla-English code-mixing asking for 3mm steel parameter numbers."
    },
    {
        "query_id": "PARAM_20",
        "text": "What heat input should I target for 8 mm mild steel MIG welding?",
        "expected_intent": "param",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Targets heat input setpoint for 8 mm mild steel."
    },

    # =========================================================================
    # INTENT: knowledge (20 queries)
    # =========================================================================
    {
        "query_id": "KNOW_01",
        "text": "What causes porosity in MIG welding and how do I prevent it?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "General welding defect knowledge query."
    },
    {
        "query_id": "KNOW_02",
        "text": "How do I fix undercut at the weld toe?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Welding defect remediation troubleshooting."
    },
    {
        "query_id": "KNOW_03",
        "text": "What shielding gas mixture is recommended for stainless steel GMAW?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Shielding gas metallurgical selection inquiry."
    },
    {
        "query_id": "KNOW_04",
        "text": "reduce spatter",
        "expected_intent": "knowledge",
        "perturbation_type": "short_query",
        "difficulty": "easy",
        "rationale": "Two-word defect reduction query."
    },
    {
        "query_id": "KNOW_05",
        "text": "avoid lack of fusion",
        "expected_intent": "knowledge",
        "perturbation_type": "short_query",
        "difficulty": "medium",
        "rationale": "Brief defect prevention request."
    },
    {
        "query_id": "KNOW_06",
        "text": "Why do wormholes and trapped gas bubbles form inside the solidifying molten weld pool?",
        "expected_intent": "knowledge",
        "perturbation_type": "paraphrase",
        "difficulty": "medium",
        "rationale": "Paraphrase of porosity using metallurgical terminology."
    },
    {
        "query_id": "KNOW_07",
        "text": "My beads look like ugly rope and won't wet into the sides at all.",
        "expected_intent": "knowledge",
        "perturbation_type": "informal_slang",
        "difficulty": "medium",
        "rationale": "Operator slang describing narrow bead / poor wetting."
    },
    {
        "query_id": "KNOW_08",
        "text": "how to stop procity and spater in mig",
        "expected_intent": "knowledge",
        "perturbation_type": "typos_spelling",
        "difficulty": "hard",
        "rationale": "Typos in porosity ('procity') and spatter ('spater')."
    },
    {
        "query_id": "KNOW_09",
        "text": "Difference between GMAW and FCAW in deposition rate?",
        "expected_intent": "knowledge",
        "perturbation_type": "abbreviations",
        "difficulty": "medium",
        "rationale": "Process acronyms GMAW and FCAW compared."
    },
    {
        "query_id": "KNOW_10",
        "text": "weld e excessive spatter hocche kivabe thik korbo?",
        "expected_intent": "knowledge",
        "perturbation_type": "code_switching",
        "difficulty": "medium",
        "rationale": "Code-switched query asking how to fix excessive weld spatter."
    },
    {
        "query_id": "KNOW_11",
        "text": "In stainless steel pipe welding why is root back-purging necessary?",
        "expected_intent": "knowledge",
        "perturbation_type": "word_order",
        "difficulty": "medium",
        "rationale": "Metallurgical question on back-purging and sugaring."
    },
    {
        "query_id": "KNOW_12",
        "text": "We are having discussions on the factory floor and I was wondering if you could explain what causes delayed cracking after cooling.",
        "expected_intent": "knowledge",
        "perturbation_type": "noisy_redundant",
        "difficulty": "medium",
        "rationale": "Conversational preamble surrounding hydrogen cold cracking."
    },
    {
        "query_id": "KNOW_13",
        "text": "How does excessive current cause both undercut and burn through?",
        "expected_intent": "knowledge",
        "perturbation_type": "multi_keyword",
        "difficulty": "hard",
        "rationale": "Multi-defect comparison linking current to undercut and burn through."
    },
    {
        "query_id": "KNOW_14",
        "text": "burn-through causes",
        "expected_intent": "knowledge",
        "perturbation_type": "short_ambiguous",
        "difficulty": "medium",
        "rationale": "Hyphenated defect keyword inquiry."
    },
    {
        "query_id": "KNOW_15",
        "text": "What is the physical mechanism of spray transfer in GMAW?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Metal transfer mode physics inquiry."
    },
    {
        "query_id": "KNOW_16",
        "text": "How do I calculate heat input from current, voltage, and travel speed?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Welding heat input equation inquiry."
    },
    {
        "query_id": "KNOW_17",
        "text": "How does Isaac Sim model robotic seam tracking using force/torque sensors?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Robotics simulation domain knowledge."
    },
    {
        "query_id": "KNOW_18",
        "text": "What are the four primary levers to improve welding productivity?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Welding shop economics and efficiency inquiry."
    },
    {
        "query_id": "KNOW_19",
        "text": "Why should CO2 content in stainless shielding gas be kept below 2 percent?",
        "expected_intent": "knowledge",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Corrosion resistance and carbide precipitation inquiry."
    },
    {
        "query_id": "KNOW_20",
        "text": "explain wps compliance",
        "expected_intent": "knowledge",
        "perturbation_type": "short_query",
        "difficulty": "easy",
        "rationale": "Short query on Welding Procedure Specification."
    },

    # =========================================================================
    # INTENT: general (20 queries)
    # =========================================================================
    {
        "query_id": "GEN_01",
        "text": "What causes chatter in CNC milling and how do I reduce it?",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Machining domain query matching 'cnc' and 'milling'."
    },
    {
        "query_id": "GEN_02",
        "text": "How do I prevent sink marks in injection molding of thick plastic parts?",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Polymer processing domain matching 'injection molding' and 'sink mark'."
    },
    {
        "query_id": "GEN_03",
        "text": "Explain the difference between sand casting and investment casting.",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Foundry/casting process comparison matching 'casting'."
    },
    {
        "query_id": "GEN_04",
        "text": "cnc chatter",
        "expected_intent": "general",
        "perturbation_type": "short_query",
        "difficulty": "easy",
        "rationale": "Two-word machining query matching CNC and chatter."
    },
    {
        "query_id": "GEN_05",
        "text": "short shot in molding",
        "expected_intent": "general",
        "perturbation_type": "short_query",
        "difficulty": "medium",
        "rationale": "Brief defect inquiry for injection molding."
    },
    {
        "query_id": "GEN_06",
        "text": "What steps should be taken to eliminate warpage on thin FDM 3D printed ABS parts?",
        "expected_intent": "general",
        "perturbation_type": "paraphrase",
        "difficulty": "medium",
        "rationale": "Additive manufacturing inquiry matching '3d print' and 'fdm'."
    },
    {
        "query_id": "GEN_07",
        "text": "The lathe is squealing like crazy and leaving a rough finish on the shaft.",
        "expected_intent": "general",
        "perturbation_type": "informal_slang",
        "difficulty": "medium",
        "rationale": "Colloquial shop description matching 'lathe'."
    },
    {
        "query_id": "GEN_08",
        "text": "how to calculate oee for assemly line",
        "expected_intent": "general",
        "perturbation_type": "typos_spelling",
        "difficulty": "hard",
        "rationale": "Typos in assembly while matching OEE."
    },
    {
        "query_id": "GEN_09",
        "text": "How do CMM and GD&T tolerances improve quality control?",
        "expected_intent": "general",
        "perturbation_type": "abbreviations",
        "difficulty": "medium",
        "rationale": "Metrology acronyms CMM and GD&T."
    },
    {
        "query_id": "GEN_10",
        "text": "cnc lathe e cutting speed koto rakhte hobe?",
        "expected_intent": "general",
        "perturbation_type": "code_switching",
        "difficulty": "medium",
        "rationale": "Bangla-English code-switching for CNC lathe cutting speed."
    },
    {
        "query_id": "GEN_11",
        "text": "In sheet metal press brake bending how do you compensate for elastic springback?",
        "expected_intent": "general",
        "perturbation_type": "word_order",
        "difficulty": "medium",
        "rationale": "Forming process inquiry matching 'press brake' and 'springback'."
    },
    {
        "query_id": "GEN_12",
        "text": "Our continuous improvement team wants to know the primary differences between kaizen 5S and kanban systems.",
        "expected_intent": "general",
        "perturbation_type": "noisy_redundant",
        "difficulty": "medium",
        "rationale": "Lean manufacturing methodology matching 'kaizen', '5s', 'kanban'."
    },
    {
        "query_id": "GEN_13",
        "text": "Compare CNC milling cycle time against die casting for high-volume aluminum parts.",
        "expected_intent": "general",
        "perturbation_type": "multi_keyword",
        "difficulty": "hard",
        "rationale": "Cross-process comparison matching CNC, milling, and die casting."
    },
    {
        "query_id": "GEN_14",
        "text": "takt time formula",
        "expected_intent": "general",
        "perturbation_type": "short_ambiguous",
        "difficulty": "easy",
        "rationale": "Operations management concept matching 'takt time'."
    },
    {
        "query_id": "GEN_15",
        "text": "What is the difference between annealing and tempering in heat treatment?",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Metallurgical heat treatment process matching 'annealing' and 'tempering'."
    },
    {
        "query_id": "GEN_16",
        "text": "How do I choose the correct clamp tonnage for an injection molding machine?",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Molding machine sizing matching 'clamp tonnage'."
    },
    {
        "query_id": "GEN_17",
        "text": "What causes burrs in sheet metal stamping and blanking dies?",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Stamping and tooling wear matching 'stamping' and 'blanking'."
    },
    {
        "query_id": "GEN_18",
        "text": "What spindle speed and feed rate should I use for drilling titanium on a CNC?",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "CNC drilling parameters matching 'cnc' and 'spindle speed'."
    },
    {
        "query_id": "GEN_19",
        "text": "Explain poka yoke error proofing with factory floor examples.",
        "expected_intent": "general",
        "perturbation_type": "clean_direct",
        "difficulty": "easy",
        "rationale": "Quality methodology matching 'poka yoke'."
    },
    {
        "query_id": "GEN_20",
        "text": "3d printing infill patterns comparison",
        "expected_intent": "general",
        "perturbation_type": "short_query",
        "difficulty": "easy",
        "rationale": "Additive manufacturing inquiry matching '3d printing' and 'infill'."
    }
]


def get_benchmark_queries() -> List[Dict[str, Any]]:
    """Return the 100 hand-curated intent stress benchmark queries."""
    return list(BENCHMARK_QUERIES)


def validate_benchmark_schema(queries: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validate that benchmark queries adhere to the existing intent taxonomy
    and required schema fields.
    """
    errors: List[str] = []
    seen_ids: Set[str] = set()

    for idx, q in enumerate(queries):
        qid = q.get("query_id")
        if not qid:
            errors.append(f"Query index {idx} has missing or empty query_id")
        elif qid in seen_ids:
            errors.append(f"Duplicate query_id: {qid}")
        else:
            seen_ids.add(qid)

        text = q.get("text", "")
        if not text or not text.strip():
            errors.append(f"Query {qid} has empty text")

        intent = q.get("expected_intent")
        if intent not in INTENT_TAXONOMY:
            errors.append(f"Query {qid} has invalid expected_intent '{intent}'. Must be in {INTENT_TAXONOMY}")

        ptype = q.get("perturbation_type")
        if ptype not in PERTURBATION_TYPES:
            errors.append(f"Query {qid} has invalid perturbation_type '{ptype}'. Must be in {PERTURBATION_TYPES}")

        diff = q.get("difficulty")
        if diff not in DIFFICULTY_LEVELS:
            errors.append(f"Query {qid} has invalid difficulty '{diff}'. Must be in {DIFFICULTY_LEVELS}")

        rationale = q.get("rationale")
        if not rationale or len(rationale.strip()) < 5:
            errors.append(f"Query {qid} has missing or too short rationale")

    return len(errors) == 0, errors


def save_benchmark_queries(dest_path: Optional[Path] = None) -> Path:
    """Save the benchmark queries to JSON."""
    if dest_path is None:
        dest_path = _ROOT / "evaluation" / "artifacts" / "intent_stress_benchmark.json"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_QUERIES, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return dest_path


if __name__ == "__main__":
    is_valid, errors = validate_benchmark_schema(BENCHMARK_QUERIES)
    if not is_valid:
        print(f"Validation failed with {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        out = save_benchmark_queries()
        print(f"Successfully validated and saved {len(BENCHMARK_QUERIES)} stress queries to {out}")
