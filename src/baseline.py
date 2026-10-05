import re


# Different date formats
DATE = (
    r"(?:\d{2}[./-]\d{2}[./-]\d{2,4}"
    r"|\d{2}-[A-Za-z]{3}-\d{4}"
    r"|\d{4}-\d{2}-\d{2})"
)


# Each label has one or more patterns.
PII_PATTERNS = {
    "EMAIL": [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ],
    "PATIENT_ID": [
        r"\b(?:B-\d{6}|CHN-\d{7}|HYD\d{6}|BER/\d{4}/\d{2}|UHID/\d{5}/\d{2}|MRN-\d{2}-\d{5})\b",
    ],
    "DATE_OF_BIRTH": [
        rf"(?i)(?:\bDOB\b|date of birth)\s*[:=]?\s*({DATE})",
    ],
    "ENCOUNTER_DATE": [
        rf"(?i)(?:visit|admission|encounter date|DOA)\s*[:=]?\s*({DATE})",
    ],
    "PHONE_NUMBER": [
        r"(?i)(?:telephone|contact|mobile|phone|\bph)\s*[:=]?\s*(\+?\d[\d ()-]{8,}\d)",
    ],
    "ADDRESS": [
        
        # The contact heading can be written with or without ":".
        r"(?i)(?:address|residence|home)\s*[:=]?\s*(.+?)(?=\s*(?:\||;|\n|(?:telephone|mobile|contact|phone|ph|treating clinician|attending physician|consultant|reviewed by|author|signed(?: electronically)?(?: by)?|responsible doctor|clinical data|laboratory|echocardiography|diagnosis|dx|medications?|rx|allergy|smoking)\s*[:=]?))",
        # Compact notes can use "from Flat ...".
        r"(?i)\bfrom\s+(Flat\s+[^;\n|.]+)",
    ],
    "PATIENT_NAME": [
        r"(?i)(?:patient|name|\bpt)\s*[:=:]?\s*([A-Z][A-Za-z'-]+\s+[A-Z][A-Za-z'-]+)",
    ],
    "CLINICIAN_NAME": [
        r"(?i)(?:treating clinician|attending physician|consultant|reviewed by|author|signed(?: electronically)?(?: by)?|responsible doctor)\s*[:=]?\s*(?:Dr\.?\s*)?([A-Z][\w'-]+\s+[A-Z][\w'-]+)",
    ],
}


# Different spellings and meanings for diagnoses.
DIAGNOSES = {
    "atrial_fibrillation": (
        "atrial fibrillation",
        r"\baf\b",
        r"\ba-fib\b",
        r"\bafib\b",
        "vorhofflimmern",
    ),
    "heart_failure": (
        "heart failure",
        r"\bhfref\b",
        r"\bhfpef\b",
        "cardiac failure",
        "chronic cardiac failure",
        r"\blv failure\b",
    ),
    "hypertension": (
        "hypertension",
        "arterial hypertension",
        r"\bhtn\b",
        "hypertensive disease",
        "systemic hypertension",
    ),
    "type_2_diabetes": (
        "type 2 diabetes",
        "type ii diabetes",
        "diabetes mellitus",
        r"\bt2dm\b",
        r"\bdm2\b",
        r"type\s*2\s*dm\b",
    ),
    "chronic_kidney_disease": (
        "chronic kidney disease",
        r"\bckd\b",
        "chronic renal disease",
        "chronic renal dysfunction",
    ),
    "coronary_artery_disease": (
        "coronary artery disease",
        "ischemic heart disease",
        "ischaemic heart disease",
        "coronary disease",
        r"\bcad\b",
        r"\bihd\b",
    ),
    "acute_coronary_syndrome": (
        "acute coronary syndrome",
        r"\bacs\b",
        "unstable angina",
    ),
    "pneumonia": (
        "pneumonia",
        "infective consolidation",
        r"\bcap\b",
        "community acquired pneumonia",
    ),
    "copd": (
        "chronic obstructive pulmonary disease",
        r"\bcopd\b",
    ),
}


# Brand names are mapped to the required generic medication names.
MEDICATIONS = {
    "apixaban": ("apixaban", "eliquis"),
    "rivaroxaban": ("rivaroxaban", "xarelto"),
    "warfarin": ("warfarin", "coumadin"),
    "metoprolol": ("metoprolol", "lopressor", r"toprol(?:-xl)?"),
    "bisoprolol": ("bisoprolol", "concor"),
    "furosemide": ("furosemide", "frusemide", "lasix"),
    "ramipril": ("ramipril", "altace"),
    "amlodipine": ("amlodipine", "norvasc"),
    "metformin": ("metformin", "glucophage"),
    "insulin": ("insulin",),
    "atorvastatin": ("atorvastatin", "lipitor"),
    "aspirin": (
        "aspirin",
        "ecosprin",
        "acetylsalicylic acid",
        r"\basa\b",
    ),
    "clopidogrel": ("clopidogrel", "plavix"),
    "amiodarone": ("amiodarone", "cordarone"),
    "digoxin": ("digoxin", "lanoxin"),
    "azithromycin": ("azithromycin", "azithro", "zithromax"),
}


def add_span(spans, start, end, label):
    """Add a PII span only if it is valid and does not overlap another span."""
    if start < 0 or end <= start:
        return

    for old_span in spans:
        overlaps = start < old_span["end"] and end > old_span["start"]

        if overlaps:
            return

    spans.append({
        "start": start,
        "end": end,
        "label": label,
    })


def detect_pii(note):
    """Find PII entities and return non-overlapping character spans."""
    spans = []

    # First, find ordinary patterns with explicit headings.
    for label, patterns in PII_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, note):
                if match.lastindex:
                    start, end = match.span(1)
                else:
                    start, end = match.span()

                add_span(spans, start, end, label)

    # Compact record example:
    # fot example : "Keerthi Gupta / HYD439096 / born 08/11/1982"
    compact_name = (
        r"(?im)(?:^|\b(?:note|record|summary)\s+)"
        r"([A-Z][\w'-]+\s+[A-Z][\w'-]+)"
        r"\s*/\s*(?:HYD|CHN|BER)"
    )

    for match in re.finditer(compact_name, note):
        add_span(spans, *match.span(1), "PATIENT_NAME")

    # Another compact record example: "Anton Adler, born 23.05.1963"
   
    born_name = r"(?i)\b([A-Z][\w'-]+\s+[A-Z][\w'-]+),\s*born\b"

    for match in re.finditer(born_name, note):
        add_span(spans, *match.span(1), "PATIENT_NAME")

    # If a date is directly after "DOB" or "born", it is a date of birth :)
    for match in re.finditer(DATE, note):
        text_before = note[max(0, match.start() - 35):match.start()].lower()

        if re.search(r"(?:dob|born)\s*$", text_before):
            label = "DATE_OF_BIRTH"
        else:
            label = "ENCOUNTER_DATE"

        add_span(spans, *match.span(), label)

    # A doctor name often appears in both the clinician field and signature.
    clinician_names = []

    for span in spans:
        if span["label"] == "CLINICIAN_NAME":
            name = note[span["start"]:span["end"]]
            clinician_names.append(name)

    for name in clinician_names:
        for match in re.finditer(re.escape(name), note):
            add_span(spans, *match.span(), "CLINICIAN_NAME")

    # Remove punctuation or whitespace accidentally captured after addresses.
    for span in spans:
        if span["label"] == "ADDRESS":
            while (
                span["end"] > span["start"]
                and note[span["end"] - 1] in " .,:;"
            ):
                span["end"] -= 1

    return sorted(spans, key=lambda item: (item["start"], item["end"]))



def render_deidentified(note, spans):
    """Replace PII spans with label placeholders."""
    # Replace from right to left so original offsets remain correct.
    for span in reversed(spans):
        replacement = f"[{span['label']}]"

        note = (
            note[:span["start"]]
            + replacement
            + note[span["end"]:]
        )

    return note


def find_number(pattern, text):
    """Return the first number matched by a regex, or None."""
    match = re.search(pattern, text, re.IGNORECASE)

    if match is None:
        return None

    # Support decimal commas such as "1,20".
    return float(match.group(1).replace(",", "."))


def is_active_diagnosis(note, pattern):
    """Return True when an alias appears without a nearby negation."""
    for match in re.finditer(pattern, note, re.IGNORECASE):
        text_before = note[max(0, match.start() - 35):match.start()].lower()
# in a real world we need a stronger NLP/NER and negation detection
        negative_words = (
            "no evidence of",
            "no ",
            "denies",
            "ruled out",
            "family history of",
        )

        if not any(word in text_before for word in negative_words):
            return True

    return False


def find_terms(note, dictionary, check_negation=False):
    """Find canonical diagnosis or medication terms from alias dictionaries."""
    results = []

    for standard_name, aliases in dictionary.items():
        for alias in aliases:
            found = re.search(alias, note, re.IGNORECASE)

            if found is None:
                continue

            if check_negation and not is_active_diagnosis(note, alias):
                continue

            results.append(standard_name)
            break

    return sorted(results)


def find_smoking_status(note):
    """Return never, former, current, or None."""
    note = note.lower()

    if any(word in note for word in (
        "never smoked",
        "non-smoker",
        "no tobacco use",
    )):
        return "never"

    if any(word in note for word in (
        "former smoker",
        "ex-smoker",
        "stopped smoking",
    )):
        return "former"

    if any(word in note for word in (
        "current smoker",
        "ongoing tobacco",
        "actively smokes",
    )):
        return "current"

    return None


def find_allergy(note):
    """Return the required canonical allergy name, or None."""
    note = note.lower()

    if any(word in note for word in (
        "nkda",
        "no known drug allergies",
        "no medication allergy",
    )):
        return "none"

    if "penicillin" in note:
        return "penicillin"

    if any(word in note for word in (
        "nsaid",
        "ibuprofen",
        "non-steroidal",
    )):
        return "nsaid"

    if "contrast" in note:
        return "iodinated_contrast"

    return None


def extract_clinical_data(note):
    """Extract all fields required by the challenge submission schema."""
    heart_rate = find_number(
        r"(?:\bhr\b|pulse|ventricular rate)\s*[=:]?\s*(\d{2,3})",
        note,
    )

    systolic_bp = find_number(
        r"(?:\bbp\b|blood pressure)\s*[=:]?\s*(\d{2,3})(?:\s*/|\s+over)",
        note,
    )

    creatinine_mg_dl = find_number(
        r"(?:serum\s+)?creatinine\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*mg/dl",
        note,
    )

    creatinine_umol_l = find_number(
        r"(?:serum\s+)?creatinine\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*(?:µmol/l|umol/l)",
        note,
    )

    hemoglobin_g_dl = find_number(
        r"(?:hemoglobin|\bhb\b)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*g/dl",
        note,
    )

    hemoglobin_g_l = find_number(
        r"(?:hemoglobin|\bhb\b)\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*g/l",
        note,
    )

    lvef = find_number(
        r"(?:lvef|\bef\b|ejection fraction(?: approximately)?)\s*[=:]?\s*(\d{2})",
        note,
    )

    # Convert alternative laboratory units to required units.
    if creatinine_mg_dl is None and creatinine_umol_l is not None:
        creatinine_mg_dl = round(creatinine_umol_l / 88.4, 2)

    if hemoglobin_g_dl is None and hemoglobin_g_l is not None:
        hemoglobin_g_dl = round(hemoglobin_g_l / 10, 2)

    return {
        "diagnoses": find_terms(
            note,
            DIAGNOSES,
            check_negation=True,
        ),
        "medications": find_terms(note, MEDICATIONS),
        "heart_rate_bpm": int(heart_rate) if heart_rate is not None else None,
        "systolic_bp_mmhg": (
            int(systolic_bp)
            if systolic_bp is not None
            else None
        ),
        "creatinine_mg_dl": creatinine_mg_dl,
        "hemoglobin_g_dl": hemoglobin_g_dl,
        "lvef_percent": int(lvef) if lvef is not None else None,
        "smoking_status": find_smoking_status(note),
        "allergy": find_allergy(note),
    }
