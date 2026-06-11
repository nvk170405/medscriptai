"""Medical vocabulary — common Indian drug names, dosages, and frequencies."""

from __future__ import annotations


# ── Common Indian Prescription Drug Names ────────────────────────────────────
# Sourced from Indian pharmacopoeia and common OTC drugs

DRUG_NAMES: list[str] = [
    # Antibiotics
    "Amoxicillin", "Azithromycin", "Cephalexin", "Ciprofloxacin", "Doxycycline",
    "Erythromycin", "Levofloxacin", "Metronidazole", "Norfloxacin", "Ofloxacin",
    "Penicillin", "Tetracycline", "Cefixime", "Ceftriaxone", "Clindamycin",
    "Amoxyclav", "Augmentin", "Cefpodoxime", "Nitrofurantoin", "Cotrimoxazole",

    # Analgesics & Anti-inflammatory
    "Paracetamol", "Ibuprofen", "Diclofenac", "Aspirin", "Naproxen",
    "Aceclofenac", "Piroxicam", "Mefenamic", "Tramadol", "Dolo",
    "Crocin", "Combiflam", "Voveran", "Sumo", "Flexon",

    # Gastrointestinal
    "Omeprazole", "Pantoprazole", "Ranitidine", "Domperidone", "Ondansetron",
    "Rabeprazole", "Esomeprazole", "Sucralfate", "Metoclopramide", "Loperamide",
    "Antacid", "Digene", "Gelusil", "Rantac", "Pan",

    # Cardiovascular
    "Amlodipine", "Atenolol", "Metoprolol", "Losartan", "Telmisartan",
    "Enalapril", "Ramipril", "Clopidogrel", "Atorvastatin", "Rosuvastatin",
    "Ecosprin", "Cardivas", "Concor", "Telma", "Olmesartan",

    # Antidiabetics
    "Metformin", "Glimepiride", "Glipizide", "Sitagliptin", "Vildagliptin",
    "Gliclazide", "Pioglitazone", "Insulin", "Glycomet", "Januvia",

    # Respiratory
    "Montelukast", "Salbutamol", "Budesonide", "Cetirizine", "Levocetirizine",
    "Fexofenadine", "Loratadine", "Theophylline", "Chlorpheniramine", "Dextromethorphan",
    "Ambroxol", "Bromhexine", "Guaifenesin", "Deriphyllin", "Asthalin",

    # Corticosteroids
    "Prednisolone", "Dexamethasone", "Hydrocortisone", "Betamethasone",
    "Methylprednisolone", "Deflazacort", "Budesonide", "Wysolone",

    # Vitamins & Supplements
    "Multivitamin", "Calcium", "Iron", "Folic", "Vitamin",
    "Zinc", "Calcitriol", "Cholecalciferol", "Shelcal", "Becosules",
    "Neurobion", "Zincovit", "Revital", "Livogen", "Ferrous",

    # Antifungal
    "Fluconazole", "Clotrimazole", "Ketoconazole", "Itraconazole", "Terbinafine",

    # Psychiatric / CNS
    "Alprazolam", "Diazepam", "Clonazepam", "Sertraline", "Escitalopram",
    "Fluoxetine", "Amitriptyline", "Gabapentin", "Pregabalin", "Carbamazepine",
    "Phenytoin", "Valproate", "Lithium", "Olanzapine", "Risperidone",

    # Dermatology
    "Clobetasol", "Mupirocin", "Fusidic", "Permethrin", "Ivermectin",

    # Ophthalmology
    "Tobramycin", "Moxifloxacin", "Timolol", "Latanoprost",

    # Common Indian brand names
    "Calpol", "Mox", "Zifi", "Monocef", "Taxim",
    "Wikoryl", "Sinarest", "Allegra", "Montair", "Alex",
]

# ── Dosage Forms ─────────────────────────────────────────────────────────────

DOSAGE_FORMS: list[str] = [
    "Tablet", "Tab", "Capsule", "Cap", "Syrup", "Syr",
    "Injection", "Inj", "Cream", "Ointment", "Drops",
    "Suspension", "Susp", "Gel", "Lotion", "Spray",
    "Inhaler", "Patch", "Suppository", "Sachet", "Powder",
]

# ── Dosage Units ─────────────────────────────────────────────────────────────

DOSAGE_UNITS: list[str] = [
    "mg", "mcg", "g", "ml", "IU", "units",
    "5mg", "10mg", "20mg", "25mg", "40mg", "50mg",
    "100mg", "150mg", "200mg", "250mg", "300mg", "400mg",
    "500mg", "625mg", "650mg", "750mg", "1000mg", "1g",
    "2.5ml", "5ml", "10ml", "15ml",
]

# ── Frequency Terms ──────────────────────────────────────────────────────────

FREQUENCY_TERMS: list[str] = [
    # Latin abbreviations
    "OD", "BD", "TDS", "TID", "QID", "QD",
    "BID", "SOS", "PRN", "HS", "AC", "PC",
    "STAT", "QHS",

    # English
    "once daily", "twice daily", "thrice daily",
    "three times a day", "twice a day", "once a day",
    "four times a day", "every 6 hours", "every 8 hours",
    "every 12 hours", "at bedtime", "before meals",
    "after meals", "before food", "after food",
    "morning", "evening", "night",
    "empty stomach", "with food",

    # Hindi transliterations
    "subah", "dopahar", "shaam", "raat",
    "khana khane se pehle", "khana khane ke baad",
]

# ── Duration Terms ───────────────────────────────────────────────────────────

DURATION_TERMS: list[str] = [
    "days", "weeks", "months", "day", "week", "month",
    "1 day", "2 days", "3 days", "5 days", "7 days",
    "10 days", "14 days", "15 days", "21 days", "30 days",
    "1 week", "2 weeks", "3 weeks", "4 weeks",
    "1 month", "2 months", "3 months", "6 months",
    "for", "x", "continue", "until review",
]

# ── Instruction Terms ────────────────────────────────────────────────────────

INSTRUCTION_TERMS: list[str] = [
    "Take", "Apply", "Instill", "Inhale", "Chew",
    "Dissolve", "Gargle", "Insert", "Inject",
    "with water", "with milk", "with food",
    "on affected area", "as directed", "as needed",
    "do not crush", "do not chew",
    "avoid alcohol", "avoid sunlight",
    "complete the course", "review after",
    "follow up", "SOS", "if needed",
    "Rx", "Sig", "Disp", "Refill",
]


# ── Lookup Helpers ───────────────────────────────────────────────────────────


def get_full_vocabulary() -> list[str]:
    """Return complete vocabulary for word beam search."""
    vocab = set()
    for word_list in [
        DRUG_NAMES, DOSAGE_FORMS, DOSAGE_UNITS,
        FREQUENCY_TERMS, DURATION_TERMS, INSTRUCTION_TERMS,
    ]:
        for term in word_list:
            # Add individual words for multi-word terms
            for word in term.split():
                vocab.add(word.lower())
                vocab.add(word)  # Keep original case too
    return sorted(vocab)


def normalize_drug_name(name: str) -> str | None:
    """
    Fuzzy-match a drug name against known drugs.

    Returns the canonical name if a close match is found, else None.
    """
    import editdistance

    name_lower = name.lower().strip()
    best_match: str | None = None
    best_distance = float("inf")

    for drug in DRUG_NAMES:
        dist = editdistance.eval(name_lower, drug.lower())
        # Allow up to 2 character edits for short names, 3 for longer
        max_dist = 2 if len(drug) <= 6 else 3
        if dist < best_distance and dist <= max_dist:
            best_distance = dist
            best_match = drug

    return best_match


def get_drug_names_set() -> set[str]:
    """Return drug names as a lowercase set for fast lookup."""
    return {d.lower() for d in DRUG_NAMES}
