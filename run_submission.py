
"""Train challenge models, create predictions, and save reports."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.baseline import detect_pii, extract_clinical_data, render_deidentified
from src.federated import (
    FEATURE_NAMES,
    SITES,
    dp_accounting,
    fit_centralized,
    fit_fedavg,
    fit_local,
    model_report,
    predict_probability,
)

SEEDS = (7, 19, 43)
DP_CLIP_NORM = 0.5
DP_NOISE_MULTIPLIER = 4.0
DP_DELTA = 1e-5


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def group_by_site(records):
    groups = {site: [] for site in SITES}

    for record in records:
        site = record.get("hospital_id")
        if site in groups:
            groups[site].append(record)

    return groups


def add_validation_labels(inputs, truth_path):
    """Attach public labels only for reporting."""

    if truth_path is None:
        return None

    labels = {
        row["case_id"]: int(row["readmission_30d"])
        for row in read_jsonl(truth_path)
    }

    labelled_records = []

    for record in inputs:
        case_id = record["case_id"]

        if case_id not in labels:
            raise ValueError(f"Missing public label for {case_id}")

        labelled_records.append(
            {
                **record,
                "labels": {
                    "readmission_30d": labels[case_id],
                },
            }
        )

    return labelled_records


def train_models(train_by_site, training, inputs, rounds, local_epochs):
    """Train FedAvg and centralized models with three fixed seeds."""

    federated_runs = []
    centralized_runs = []
    seed_7_history = []

    for seed in SEEDS:
        federated_model, history = fit_fedavg(
            train_by_site,
            rounds=rounds,
            local_epochs=local_epochs,
            seed=seed,
        )

        centralized_model = fit_centralized(
            training,
            epochs=rounds * local_epochs,
            seed=seed,
        )

        federated_runs.append(predict_probability(federated_model, inputs))
        centralized_runs.append(predict_probability(centralized_model, inputs))

        if seed == 7:
            seed_7_history = history

    return {
        "seeds": list(SEEDS),
        "fed_probabilities": np.mean(federated_runs, axis=0),
        "central_probabilities": np.mean(centralized_runs, axis=0),
        "history": seed_7_history,
    }


def build_local_model_summary(train_by_site, validation_by_site, epochs):
    summaries = {}

    for site in SITES:
        training_records = train_by_site[site]
        validation_records = validation_by_site.get(site, [])
        metrics = None

        if training_records:
            model = fit_local(training_records, epochs=epochs, seed=7)

            if validation_records:
                probabilities = predict_probability(model, validation_records)
                metrics = model_report(validation_records, probabilities)

        summaries[site] = {
            "algorithm": "regularized_logistic_regression",
            "training_cases": len(training_records),
            "training_positive": sum(
                row["labels"]["readmission_30d"]
                for row in training_records
            ),
            "validation_metrics": metrics,
        }

    return summaries


def write_predictions(path, inputs, probabilities):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record, probability in zip(inputs, probabilities):
            note = record["note_text"]
            pii_entities = detect_pii(note)

            prediction = {
                "case_id": record["case_id"],
                "pii_entities": pii_entities,
                "deidentified_text": render_deidentified(note, pii_entities),
                "extracted_clinical_data": extract_clinical_data(note),
                "readmission_probability": float(
                    np.clip(probability, 0.0, 1.0)
                ),
            }

            handle.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def run_dp_prototype(train_by_site, inputs, validation, rounds, local_epochs):
    model, history = fit_fedavg(
        train_by_site,
        rounds=rounds,
        local_epochs=local_epochs,
        seed=7,
        clip_norm=DP_CLIP_NORM,
        noise_multiplier=DP_NOISE_MULTIPLIER,
    )

    probabilities = predict_probability(model, inputs)
    metrics = (
        model_report(validation, probabilities)
        if validation is not None
        else None
    )

    client_sizes = [len(train_by_site[site]) for site in SITES]
    maximum_client_weight = max(client_sizes) / sum(client_sizes)
    sensitivity = 2 * DP_CLIP_NORM * maximum_client_weight
    accounting = dp_accounting(
        rounds,
        DP_NOISE_MULTIPLIER,
        DP_DELTA,
    )

    return {
        "history": history,
        "metrics": metrics,
        "maximum_client_weight": maximum_client_weight,
        "sensitivity": sensitivity,
        "accounting": accounting,
    }


def validation_metrics(validation, trained):
    if validation is None:
        return None, None

    federated_metrics = model_report(
        validation,
        trained["fed_probabilities"],
    )
    centralized_metrics = model_report(
        validation,
        trained["central_probabilities"],
    )

    return federated_metrics, centralized_metrics


def build_experiment_summary(
    args,
    trained,
    local_models,
    train_by_site,
    federated_metrics,
    centralized_metrics,
):
    data_by_site = {}

    for site, rows in train_by_site.items():
        positive_rate = None

        if rows:
            labels = [
                row["labels"]["readmission_30d"]
                for row in rows
            ]
            positive_rate = float(np.mean(labels))

        data_by_site[site] = {
            "n": len(rows),
            "positive_rate": positive_rate,
        }

    return {
        "implementation": "numpy_logistic_regression_fedavg",
        "random_seeds": trained["seeds"],
        "feature_set": list(FEATURE_NAMES),
        "local_models": local_models,
        "federated_model": {
            "algorithm": "FedAvg",
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "client_weighting": "n_k / N",
            "validation_metrics": federated_metrics,
            "round_history_seed_7": trained["history"],
        },
        "centralized_model": {
            "algorithm": "logistic_regression",
            "validation_metrics": centralized_metrics,
            "note": (
                "This model is only a reference because it uses pooled rows."
            ),
        },
        "data_by_site": data_by_site,
        "communication": {
            "raw_rows_transferred": False,
            "client_to_server": "sample count and model update",
            "server_to_client": "current global model parameters",
        },
        "limitations": [
            "The data are synthetic and small.",
            "Validation metrics can be unstable.",
            "FedAvg alone is not a formal privacy guarantee.",
        ],
    }


def build_privacy_summary(args, dp):
    return {
        "mechanism": "Client-level differentially private FedAvg prototype",
        "implementation_status": (
            "DP is used only for the prototype. "
            "Normal predictions use non-private FedAvg."
        ),
        "protected_asset": "One hospital client update",
        "adversary": (
            "External model recipient. The aggregation server is trusted."
        ),
        "parameters": {
            "clip_norm": DP_CLIP_NORM,
            "noise_multiplier": DP_NOISE_MULTIPLIER,
            "rounds": args.dp_rounds,
            "delta": DP_DELTA,
            "maximum_client_weight": dp["maximum_client_weight"],
            "sensitivity": dp["sensitivity"],
            "epsilon_upper_bound": (
                dp["accounting"]["epsilon_upper_bound"]
            ),
        },
        "privacy_claim": (
            "The DP prototype gives an approximate client-level privacy estimate "
            "under the trusted-server assumption. "
            "It is not patient-level differential privacy."
        ),
        "utility_analysis": {
            "dp_validation_metrics": dp["metrics"],
            "dp_round_history": dp["history"],
        },
        "limitations": [
            "The server is trusted in this prototype.",
            "The DP model is not used for normal predictions.",
            "This is not secure aggregation.",
            "This is not a patient-level privacy guarantee.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--validation-ground-truth", type=Path)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--dp-rounds", type=int, default=20)

    args = parser.parse_args()

    if min(args.rounds, args.local_epochs, args.dp_rounds) < 1:
        parser.error(
            "rounds, local epochs, and DP rounds must be positive"
        )

    return args


def main():
    args = parse_args()

    training = read_jsonl(args.train)
    inputs = read_jsonl(args.input)

    if not training:
        raise ValueError("Training file is empty")

    train_by_site = group_by_site(training)

    validation = add_validation_labels(
        inputs,
        args.validation_ground_truth,
    )
    validation_by_site = (
        group_by_site(validation)
        if validation is not None
        else {}
    )

    trained = train_models(
        train_by_site,
        training,
        inputs,
        args.rounds,
        args.local_epochs,
    )

    local_models = build_local_model_summary(
        train_by_site,
        validation_by_site,
        args.rounds * args.local_epochs,
    )

    dp = run_dp_prototype(
        train_by_site,
        inputs,
        validation,
        args.dp_rounds,
        args.local_epochs,
    )

    write_predictions(
        args.output,
        inputs,
        trained["fed_probabilities"],
    )

    federated_metrics, centralized_metrics = validation_metrics(
        validation,
        trained,
    )

    experiment = build_experiment_summary(
        args,
        trained,
        local_models,
        train_by_site,
        federated_metrics,
        centralized_metrics,
    )

    privacy = build_privacy_summary(args, dp)

    write_json(
        args.artifacts_dir / "experiment_summary.json",
        experiment,
    )

    write_json(
        args.artifacts_dir / "privacy_summary.json",
        privacy,
    )


if __name__ == "__main__":
    main()