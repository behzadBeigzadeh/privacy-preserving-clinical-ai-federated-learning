# it is better than numpy.random for DF
import secrets
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# Names of the three hospitals in this challenge
SITES = (
    "BERLIN_NODE",
    "CHENNAI_NODE",
    "HYDERABAD_NODE",
)


# Names of input values used by the prediction model
FEATURE_NAMES = (
    "age_years",
    "prior_admissions_12m",
    "length_of_stay_days",
    "emergency_admission",
    "sex_female",
    "sex_male",
    "site_BERLIN_NODE",
    "site_CHENNAI_NODE",
    "site_HYDERABAD_NODE",
)


def make_feature_row(record):
    """
    Convert one patient record to numbers.

    Example:
    age 76 becomes 0.76
    emergency admission True becomes 1.0
    """

    data = record.get("structured_features", {})
    hospital = record.get("hospital_id")

    age = float(data.get("age_years", 0)) / 100
    previous_admissions = float(data.get("prior_admissions_12m", 0)) / 5
    stay_length = float(data.get("length_of_stay_days", 0)) / 30

    emergency = float(
        bool(data.get("emergency_admission", False))
    )

    is_female = float(
        str(data.get("sex", "")).lower() == "female"
    )

    is_male = float(
        str(data.get("sex", "")).lower() == "male"
    )

    is_berlin = float(hospital == "BERLIN_NODE")
    is_chennai = float(hospital == "CHENNAI_NODE")
    is_hyderabad = float(hospital == "HYDERABAD_NODE")

    return [
        age,
        previous_admissions,
        stay_length,
        emergency,
        is_female,
        is_male,
        is_berlin,
        is_chennai,
        is_hyderabad,
    ]


def feature_matrix(records):
    """
    Convert all patient records to a NumPy matrix.

    Each row belongs to one patient.
    Each column is one feature from FEATURE_NAMES.
    """

    if not records:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)

    rows = []

    for record in records:
        row = make_feature_row(record)
        rows.append(row)

    return np.array(rows, dtype=float)


def get_labels(records):
    """
    Get readmission_30d values from labelled training records.
    """

    labels = []

    for record in records:
        label = record["labels"]["readmission_30d"]
        labels.append(int(label))

    return np.array(labels, dtype=float)


def sigmoid(values):
    """
    Convert model scores to probabilities between 0 and 1.
    """

    safe_values = np.clip(values, -30, 30)

    return 1 / (1 + np.exp(-safe_values))


def add_bias_column(features):
    """
    Add a first column containing 1.

    This helps the model learn a base probability.
    """

    bias_column = np.ones((len(features), 1))

    return np.column_stack((bias_column, features))


def create_start_weights(feature_count, seed=7):
    """
    Create small starting weights.

    A fixed seed makes normal FedAvg runs reproducible.
    """

    random = np.random.default_rng(seed)

    weights = np.zeros(feature_count + 1)

    # First item is bias, so random values start from index 1.
    weights[1:] = random.normal(
        loc=0,
        scale=0.01,
        size=feature_count,
    )

    return weights


def train_steps(
    start_weights,
    features,
    labels,
    epochs,
    learning_rate,
    l2=0.001,
    compute_loss=True,
):
    """
    Train logistic regression using simple batch gradient descent.

    start_weights:
        Current model weights.

    features:
        One row per patient.

    labels:
        Real readmission results: 0 or 1.
    """

    model = start_weights.copy()

    # Add the bias column only once before training starts.
    features_with_bias = add_bias_column(features)

    for _ in range(epochs):
        predicted_probabilities = sigmoid(
            features_with_bias @ model
        )

        errors = predicted_probabilities - labels

        gradient = (
            features_with_bias.T @ errors
        ) / len(labels)

        # Keep normal weights from becoming too large.
        # Bias is not regularized.
        gradient[1:] += l2 * model[1:]

        model -= learning_rate * gradient

    if not compute_loss:
        return model, None

    final_probabilities = sigmoid(
        features_with_bias @ model
    )

    final_probabilities = np.clip(
        final_probabilities,
        1e-8,
        1 - 1e-8,
    )

    loss = -np.mean(
        labels * np.log(final_probabilities)
        + (1 - labels) * np.log(1 - final_probabilities)
    )

    return model, float(loss)


def fit_centralized(
    records,
    epochs=2000,
    seed=7,
    learning_rate=1.0,
):
    """
    Train one reference model using pooled rows.

    This model is only for comparison with FedAvg.
    It is not the privacy-preserving workflow.
    """

    features = feature_matrix(records)
    labels = get_labels(records)

    start_weights = create_start_weights(
        len(FEATURE_NAMES),
        seed,
    )

    model, _ = train_steps(
        start_weights,
        features,
        labels,
        epochs,
        learning_rate,
    )

    return model


def fit_local(
    records,
    epochs=2000,
    seed=7,
    learning_rate=1.0,
):
    """
    Train a model using data from only one hospital.
    """

    return fit_centralized(
        records,
        epochs,
        seed,
        learning_rate,
    )


def fit_fedavg(
    client_records,
    rounds=2000,
    local_epochs=1,
    seed=7,
    clip_norm=0.5,
    noise_multiplier=None,
    learning_rate=1.0,
):
    """
    Train one global model with FedAvg.

    Each hospital trains locally.
    Only model updates are averaged by the server.
    Raw patient rows are not shared.
    """

    client_sizes = {}

    for site in SITES:
        client_sizes[site] = len(
            client_records.get(site, [])
        )

    total_records = sum(client_sizes.values())

    if total_records == 0:
        raise ValueError(
            "FedAvg needs at least one training record"
        )

    # Each hospital keeps its own features and labels.
    local_data = {}

    for site in SITES:
        records = client_records.get(site, [])

        if records:
            local_features = feature_matrix(records)
            local_labels = get_labels(records)

            local_data[site] = (
                local_features,
                local_labels,
            )

    if noise_multiplier is None:
        # Normal FedAvg has a reproducible starting point.
        global_model = create_start_weights(
            len(FEATURE_NAMES),
            seed,
        )

        random = np.random.default_rng(seed)

    else:
        # DP prototype starts from zero and uses secure random noise.
        global_model = np.zeros(
            len(FEATURE_NAMES) + 1
        )

        random = secrets.SystemRandom()

    history = []

    for round_number in range(rounds):
        average_update = np.zeros_like(global_model)
        local_losses = []

        for site in SITES:
            if site not in local_data:
                continue

            features, labels = local_data[site]

            # A hospital starts from current global model.
            local_model, loss = train_steps(
                global_model,
                features,
                labels,
                local_epochs,
                learning_rate,
                compute_loss=noise_multiplier is None,
            )

            # This update is what leaves the hospital.
            model_update = local_model - global_model

            if noise_multiplier is not None:
                update_size = np.linalg.norm(model_update)

                # Limit the size of one client update for DP.
                if update_size > clip_norm:
                    model_update *= (
                        clip_norm / update_size
                    )

            # FedAvg weights updates by hospital dataset size.
            client_weight = (
                client_sizes[site] / total_records
            )

            average_update += (
                client_weight * model_update
            )

            if noise_multiplier is None:
                local_losses.append(loss)

        if noise_multiplier is not None:
            # Add Gaussian noise only in the DP prototype.
            max_client_weight = (
                max(client_sizes.values()) / total_records
            )

            sensitivity = (
                2 * clip_norm * max_client_weight
            )

            noise_std = (
                noise_multiplier * sensitivity
            )

            noise = np.array([
                random.gauss(0, noise_std)
                for _ in range(len(global_model))
            ])

            average_update += noise

        # Server creates the next global model.
        global_model += average_update

        if noise_multiplier is None:
            history.append({
                "round": float(round_number + 1),
                "mean_client_loss": float(np.mean(local_losses)),
            })
        else:
            # Training losses are data-dependent and are not covered by the
            # update-level privacy mechanism, so never calculate or export them.
            history.append({"round": float(round_number + 1)})

    return global_model, history


def predict_probability(model, records):
    """
    Create readmission probability for each input record.
    """

    features = feature_matrix(records)
    features_with_bias = add_bias_column(features)

    return sigmoid(features_with_bias @ model)


def classification_metrics(records, probabilities):
    """
    Calculate public validation metrics.
    """

    if not records:
        return {
            "n": 0,
            "prevalence": None,
            "roc_auc": None,
            "average_precision": None,
            "brier_score": None,
            "log_loss": None,
        }

    labels = get_labels(records).astype(int)

    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7,
    )

    if len(set(labels)) == 2:
        auc = float(
            roc_auc_score(labels, probabilities)
        )
    else:
        auc = None

    if np.any(labels):
        average_precision = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )
    else:
        average_precision = None

    return {
        "n": len(labels),
        "positive": int(np.sum(labels)),
        "prevalence": float(np.mean(labels)),
        "roc_auc": auc,
        "average_precision": average_precision,
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=[0, 1],
            )
        ),
    }


def metrics_by_site(records, probabilities):
    """
    Calculate metrics separately for every hospital.
    """

    site_indexes = defaultdict(list)

    for index, record in enumerate(records):
        site = record.get("hospital_id", "UNKNOWN")
        site_indexes[site].append(index)

    result = {}

    for site, indexes in sorted(site_indexes.items()):
        site_records = []

        for index in indexes:
            site_records.append(records[index])

        site_probabilities = probabilities[indexes]

        result[site] = classification_metrics(
            site_records,
            site_probabilities,
        )

    return result


def model_report(records, probabilities):
    """
    Create one report with overall and site-level results.
    """

    return {
        "overall": classification_metrics(
            records,
            probabilities,
        ),
        "by_site": metrics_by_site(
            records,
            probabilities,
        ),
    }


def dp_accounting(rounds, noise_multiplier, delta):
    """
    Simple privacy estimate for the DP prototype.
    """

    rho = rounds / (
        2 * noise_multiplier ** 2
    )

    epsilon = rho + 2 * np.sqrt(
        rho * np.log(1 / delta)
    )

    return {
        "zcdp_rho": float(rho),
        "epsilon_upper_bound": float(epsilon),
        "delta": float(delta),
    }
