import json
import subprocess
import sys
from pathlib import Path


def test_cli_predictions_are_independent_of_evaluation_labels(tmp_path):
    root = Path(__file__).resolve().parents[1]
    truth = [
        json.loads(line)
        for line in (root / "data/validation_ground_truth.jsonl").read_text().splitlines()
    ]
    changed_truth = tmp_path / "changed_truth.jsonl"
    changed_truth.write_text(
        "".join(
            json.dumps({**row, "readmission_30d": 1 - row["readmission_30d"]}) + "\n"
            for row in truth
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(root / "run_submission.py"),
        "--train",
        str(root / "data/train.jsonl"),
        "--input",
        str(root / "data/validation_inputs.jsonl"),
        "--rounds",
        "2",
        "--local-epochs",
        "1",
        "--dp-rounds",
        "1",
    ]
    outputs = []
    for name, extra in (
        ("unlabeled", []),
        ("flipped", ["--validation-ground-truth", str(changed_truth)]),
    ):
        output = tmp_path / f"{name}.jsonl"
        subprocess.run(
            command
            + ["--output", str(output), "--artifacts-dir", str(tmp_path / name)]
            + extra,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(output.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1]
    predictions = [json.loads(line) for line in outputs[0].splitlines()]
    assert {row["case_id"] for row in predictions} == {row["case_id"] for row in truth}
    assert all(0 <= row["readmission_probability"] <= 1 for row in predictions)
