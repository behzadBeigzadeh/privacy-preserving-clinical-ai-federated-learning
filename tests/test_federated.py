import json
from pathlib import Path

import numpy as np
import pytest

from src import federated


def _record(site, index, label):
    return {
        "hospital_id": site,
        "structured_features": {
            "age_years": 40 + 10 * index,
            "sex": "male" if index % 2 else "female",
            "prior_admissions_12m": index,
            "length_of_stay_days": 3,
            "emergency_admission": bool(index % 2),
        },
        "labels": {"readmission_30d": label},
    }


def test_one_step_sample_weighted_fedavg_matches_pooled_gradient():
    rows = [
        _record("BERLIN_NODE", 0, 1),
        _record("CHENNAI_NODE", 1, 0),
        _record("CHENNAI_NODE", 2, 0),
        _record("CHENNAI_NODE", 3, 1),
    ]
    by_site = {site: [row for row in rows if row["hospital_id"] == site] for site in federated.SITES}
    fed_model, _ = federated.fit_fedavg(by_site, rounds=1, local_epochs=1, seed=11)
    pooled_model = federated.fit_centralized(rows, epochs=1, seed=11)
    np.testing.assert_allclose(fed_model, pooled_model, rtol=1e-12, atol=1e-12)


def test_dp_run_does_not_calculate_or_export_client_losses(monkeypatch):
    original = federated.train_steps
    compute_loss_flags = []

    def track_loss_setting(*args, **kwargs):
        compute_loss_flags.append(kwargs.get("compute_loss", True))
        return original(*args, **kwargs)

    monkeypatch.setattr(federated, "train_steps", track_loss_setting)
    clients = {
        "BERLIN_NODE": [_record("BERLIN_NODE", 0, 1)],
        "CHENNAI_NODE": [_record("CHENNAI_NODE", 1, 0)],
    }
    _, history = federated.fit_fedavg(
        clients,
        rounds=2,
        local_epochs=1,
        seed=7,
        clip_norm=0.5,
        noise_multiplier=4.0,
    )
    assert compute_loss_flags == [False] * 4
    assert history == [{"round": 1.0}, {"round": 2.0}]


def test_public_validation_labels_do_not_change_predictions(tmp_path):
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
        __import__("sys").executable,
        str(root / "run_submission.py"),
        "--train", str(root / "data/train.jsonl"),
        "--input", str(root / "data/validation_inputs.jsonl"),
        "--rounds", "2",
        "--local-epochs", "1",
        "--dp-rounds", "1",
    ]
    outputs = []
    import subprocess

    for name, extra in (("unlabeled", []), ("flipped", ["--validation-ground-truth", str(changed_truth)])):
        output = tmp_path / f"{name}.jsonl"
        subprocess.run(
            command + ["--output", str(output), "--artifacts-dir", str(tmp_path / name)] + extra,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(output.read_text(encoding="utf-8"))
    assert outputs[0] == outputs[1]


def test_dp_accounting_matches_gaussian_zcdp_conversion():
    result = federated.dp_accounting(rounds=20, noise_multiplier=4.0, delta=1e-5)
    assert result["epsilon_upper_bound"] == pytest.approx(5.989915065723368)
