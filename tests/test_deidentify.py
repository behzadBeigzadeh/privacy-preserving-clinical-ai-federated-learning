import json
from pathlib import Path

from src.baseline import detect_pii, render_deidentified


def test_supplied_notes_and_flattened_formats():
    data = Path(__file__).resolve().parents[1] / "data"
    training = [
        json.loads(line)
        for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (data / "validation_inputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    truth = {
        row["case_id"]: row
        for row in map(
            json.loads,
            (data / "validation_ground_truth.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    pairs = [(row, row["labels"]) for row in training] + [
        (row, truth[row["case_id"]]) for row in validation
    ]
    for row, expected in pairs:
        spans = [
            {key: span[key] for key in ("start", "end", "label")}
            for span in expected["pii_entities"]
        ]
        for flatten in (False, True):
            note = row["note_text"].replace("\n", " ") if flatten else row["note_text"]
            actual = detect_pii(note)
            assert actual == spans, row["case_id"]
            rendered = (
                expected["deidentified_text"].replace("\n", " ")
                if flatten
                else expected["deidentified_text"]
            )
            assert render_deidentified(note, actual) == rendered, row["case_id"]
