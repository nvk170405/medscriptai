"""Rule-based medical entity extractor for prescription text.

Uses regex patterns and medical vocabulary to extract structured entities
(medicines, dosages, frequencies, durations, instructions) from OCR'd text.
This is more reliable than an untrained NER model for structured prescription text.
"""

from __future__ import annotations

import re
from typing import Any

from medscript.utils.logging import get_logger

logger = get_logger(__name__)

# ── Medical vocabularies ─────────────────────────────────────────────────────

# Common medicine names (expandable)
KNOWN_MEDICINES = {
    "amoxicillin", "paracetamol", "ibuprofen", "metformin", "omeprazole",
    "pantoprazole", "atorvastatin", "amlodipine", "losartan", "metoprolol",
    "aspirin", "clopidogrel", "warfarin", "ciprofloxacin", "azithromycin",
    "doxycycline", "cetirizine", "montelukast", "salbutamol", "prednisolone",
    "prednisone", "diclofenac", "naproxen", "gabapentin", "pregabalin",
    "tramadol", "codeine", "morphine", "insulin", "glimepiride",
    "sitagliptin", "empagliflozin", "dapagliflozin", "ramipril", "enalapril",
    "lisinopril", "valsartan", "telmisartan", "hydrochlorothiazide", "furosemide",
    "spironolactone", "levothyroxine", "ranitidine", "famotidine", "esomeprazole",
    "rabeprazole", "domperidone", "ondansetron", "metoclopramide", "loperamide",
    "acetaminophen", "cephalexin", "amoxyclav", "augmentin", "cefixime",
    "ceftriaxone", "levofloxacin", "moxifloxacin", "fluconazole", "acyclovir",
    "oseltamivir", "hydroxychloroquine", "ivermectin", "albendazole",
    "multivitamin", "folic acid", "iron", "calcium", "vitamin d",
    "vitamin b12", "zinc", "vitamin c", "pantocid", "pan d", "crocin",
    "dolo", "combiflam", "saridon", "voveran", "ecosprin", "shelcal",
    "thyronorm", "glycomet", "telma", "cardace", "aten", "concor",
    "nebicord", "olmezest", "amlokind", "stamlo", "arkamin", "prazosin",
    "rosuvastatin", "fenofibrate", "crestor", "lipitor", "nexium",
    "prilosec", "prevacid", "protonix", "zantac", "pepcid",
    "benadryl", "allegra", "zyrtec", "claritin", "singulair",
    "ventolin", "symbicort", "seretide", "budesonide", "fluticasone",
}

# Dosage patterns
DOSAGE_PATTERN = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*'
    r'(mg|mcg|g|ml|iu|units?|cc|meq|mmol|%)\b',
    re.IGNORECASE,
)

# Frequency terms
FREQUENCY_PATTERNS = [
    (re.compile(r'\b(TID|t\.i\.d\.?)\b', re.IGNORECASE), "TID (three times daily)"),
    (re.compile(r'\b(BID|b\.i\.d\.?)\b', re.IGNORECASE), "BID (twice daily)"),
    (re.compile(r'\b(QID|q\.i\.d\.?)\b', re.IGNORECASE), "QID (four times daily)"),
    (re.compile(r'\b(OD|o\.d\.?|once daily)\b', re.IGNORECASE), "OD (once daily)"),
    (re.compile(r'\b(QHS|at bedtime|at night|HS|h\.s\.?)\b', re.IGNORECASE), "QHS (at bedtime)"),
    (re.compile(r'\b(SOS|as needed|PRN|p\.r\.n\.?)\b', re.IGNORECASE), "SOS (as needed)"),
    (re.compile(r'\b(BD)\b', re.IGNORECASE), "BD (twice daily)"),
    (re.compile(r'\b(AC|before meals?|before food)\b', re.IGNORECASE), "AC (before meals)"),
    (re.compile(r'\b(PC|after meals?|after food)\b', re.IGNORECASE), "PC (after meals)"),
    (re.compile(r'\btwice\s+(?:a\s+)?day\b', re.IGNORECASE), "BID (twice daily)"),
    (re.compile(r'\bthrice\s+(?:a\s+)?day\b', re.IGNORECASE), "TID (three times daily)"),
    (re.compile(r'\bonce\s+(?:a\s+)?day\b', re.IGNORECASE), "OD (once daily)"),
    (re.compile(r'\b(\d)\s*(?:times?\s+(?:a\s+)?day|x\s*(?:daily|/day))\b', re.IGNORECASE), None),
]

# Duration patterns
DURATION_PATTERN = re.compile(
    r'\b(?:for\s+)?(\d+)\s*(days?|weeks?|months?|d|w|m)\b',
    re.IGNORECASE,
)

# Instruction patterns
INSTRUCTION_PATTERNS = [
    re.compile(r'\b(with food|with meals?|after food|before food|on empty stomach)\b', re.IGNORECASE),
    re.compile(r'\b(with water|with milk)\b', re.IGNORECASE),
    re.compile(r'\b(do not crush|do not chew|swallow whole)\b', re.IGNORECASE),
    re.compile(r'\b(apply topically|apply externally|for external use)\b', re.IGNORECASE),
    re.compile(r'\b(take\s+\d+\s*tablet)\b', re.IGNORECASE),
]


def extract_entities(text: str) -> list[dict[str, Any]]:
    """
    Extract medical entities from prescription text using rules and patterns.

    Returns:
        List of entity dicts with keys: type, value, confidence
    """
    entities: list[dict[str, Any]] = []
    text_lower = text.lower()

    # 1. Extract medicines
    words = re.split(r'[\s|,;]+', text)
    for i, word in enumerate(words):
        word_clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        if word_clean in KNOWN_MEDICINES:
            # Use original casing from the text
            entities.append({
                "type": "medicine",
                "value": word,
                "confidence": 0.92,
            })
        # Also check bigrams (e.g. "folic acid", "vitamin d")
        if i < len(words) - 1:
            bigram = f"{word_clean} {re.sub(r'[^a-zA-Z]', '', words[i+1]).lower()}"
            if bigram in KNOWN_MEDICINES:
                entities.append({
                    "type": "medicine",
                    "value": f"{word} {words[i+1]}",
                    "confidence": 0.90,
                })

    # 2. Extract dosages
    for match in DOSAGE_PATTERN.finditer(text):
        entities.append({
            "type": "dosage",
            "value": match.group(0),
            "confidence": 0.95,
        })

    # 3. Extract frequencies
    for pattern, label in FREQUENCY_PATTERNS:
        match = pattern.search(text)
        if match:
            entities.append({
                "type": "frequency",
                "value": label if label else f"{match.group(0)} times/day",
                "confidence": 0.90,
            })

    # 4. Extract durations
    for match in DURATION_PATTERN.finditer(text):
        entities.append({
            "type": "duration",
            "value": match.group(0),
            "confidence": 0.88,
        })

    # 5. Extract instructions
    for pattern in INSTRUCTION_PATTERNS:
        match = pattern.search(text)
        if match:
            entities.append({
                "type": "instruction",
                "value": match.group(0),
                "confidence": 0.85,
            })

    # Deduplicate by value
    seen_values: set[str] = set()
    unique_entities: list[dict[str, Any]] = []
    for e in entities:
        val_key = f"{e['type']}:{e['value'].lower()}"
        if val_key not in seen_values:
            seen_values.add(val_key)
            unique_entities.append(e)

    logger.info("entities_extracted", count=len(unique_entities))
    return unique_entities
