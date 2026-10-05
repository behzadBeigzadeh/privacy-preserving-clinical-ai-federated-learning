
# Challenge Report

## 1. Executive summary
This solution addresses three clinical tasks:

1. Removing personally identifiable information from clinical notes
2. Extracting structured clinical information from unstructured text
3. Predicting 30-day readmission risk with federated learning

The text-processing components use transparent rule-based methods. The prediction component is a small NumPy logistic-regression model trained with Federated Averaging (FedAvg) across three hospital partitions.A separate test version was created to help protect each hospital’s data while training the model.

The public benchmark result is **37.88 / 40**:

| Task | Score |
|---|---:|
| De-identification | 15 / 15 |
| Structured extraction | 15 / 15 |
| Readmission prediction | 7.88 / 10 |
| Total | 37.88 / 40 |

The evaluation covers 30 public validation cases, including six positive readmission outcomes. These results describe performance on a small synthetic benchmark and should not be interpreted as evidence of clinical effectiveness.

## 2. System architecture

The solution is organized into four main components.

| Component | Responsibility |
|---|---|
| `run_submission.py` | Reads input data, trains models, creates predictions, and writes experiment artifacts |
| `src/baseline.py` | Detects personal identifiers, creates de-identified text, and extracts clinical fields |
| `src/federated.py` | Builds model features and implements local, centralized, federated, and private training |
| `evaluator/evaluate.py` | Scores the required prediction output against public validation labels |

The training data are divided by hospital before federated training begins. Each hospital trains on its own records. The server combines local model updates using sample-weighted FedAvg.


## 3. De-identification

The de-identification module uses pattern matching, field names, note structure, and contextual cues to identify protected information(spans).

The detector covers the sensetive data, including names, patient identifiers, dates of birth, telephone numbers, addresses, docter names, and related personal details.

Each detected entity is represented by its original character offsets. The system sorts detected spans and replaces them with the required placeholders while preserving the non-sensitive parts of the note.

The public benchmark results are:

| Metric | Score |
|---|---:|
| Entity detection | 1.0000 |
| Character-level detection | 1.0000 |
| Rendered de-identified text | 1.0000 |

The rules were designed for the note formats available in the benchmark. They may require extension for unfamiliar document layouts, spelling mistakes, unlabelled names, or real-world clinical documentation that is very important for the improvments.

## 4. Structured extraction and standardization

The extraction module converts clinical note text into the structured fields required by the submission format.

The system extracts and standardizes information such as:

- Diagnoses
- Medications
- Allergy status
- Smoking status
- Numeric laboratory or clinical values
- Hospital and admission-related information

The extractor can distinguish between an active condition and a negated condition, or between an active medication and one that has been stopped.

Missing information is represented as `null`. Supported numeric formats and units are normalized before being returned.

The public structured extraction score is `1.0000`. Because the benchmark notes are public and informed rule development, this result should be treated as a development result rather than an independent estimate of performance on unseen clinical data.

## 5. Federated-learning experiment

The readmission model is a regularized logistic-regression model implemented with NumPy.

The feature set includes:

- Age
- Sex
- Number of previous admissions
- Length of stay
- Emergency admission status
- Hospital indicator variables

The training dataset contains 120 records distributed across three hospital partitions:

| Hospital | Training records |
|---|---:|
| Berlin | 42 |
| Chennai | 39 |
| Hyderabad | 39 |

The federated model uses FedAvg. During each communication round, every hospital performs local training and sends a model update to the central coordinator. The coordinator averages client updates in proportion to each hospital's number of training records.

The standard configuration uses:

| Setting | Value |
|---|---:|
| FedAvg rounds | 2,000 |
| Local epochs per round | 1 |
| Learning rate | 1.0 |
| L2 regularization | 0.001 |
| Random seeds | 7, 19, 43 |

Three fixed training seeds are used. The final prediction probability is the average of the three federated model outputs.

A centralized model is also trained as a reference. It uses pooled training records and is not used as the final submission model.

Public validation results for the final federated prediction are:

| Metric | Value |
|---|---:|
| Readmission prediction score | 0.7876 |
| ROC-AUC | 0.7986 |
| Average precision | 0.7107 |
| Brier score | 0.1467 |
| Log loss | 0.4714 |

The validation set is small. Each hospital has only ten validation records, and the Hyderabad partition has only one positive outcome. Therefore, hospital-level metrics should be interpreted carefully.

## 6. Privacy extension and threat model

The project includes a separate client-level differentially private FedAvg prototype.

The protected unit is one hospital client update. The private prototype applies the following steps:

1. Clip each client update to a fixed norm
2. Combine updates using fixed client weights
3. Add Gaussian noise to the aggregated update
4. Use zCDP accounting to estimate the privacy budget

The private prototype uses:

| Parameter | Value |
|---|---:|
| Clip norm | 0.5 |
| Noise multiplier | 4.0 |
| Private rounds | 20 |
| Delta | 1e-5 |
| Approximate epsilon | 5.9899 |

The server can observe individual client updates before aggregation. Secure aggregation is not implemented.

The private training version is separate from the main model. The final predictions use normal FedAvg, not differential privacy.  

The private version does not save each hospital’s training loss. It is only a simple example to show how privacy can work during training.

## 7. Reproducibility

Run the following command from the repository root to generate predictions and experiment artifacts:

```powershell
python run_submission.py --train data/train.jsonl --input data/validation_inputs.jsonl --output outputs/validation_predictions.jsonl --artifacts-dir outputs/artifacts
```

Run the evaluator with:

```powershell
python evaluator/evaluate.py --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --predictions outputs/validation_predictions.jsonl --report outputs/validation_report.json
```

The main output files are:

| File | Description |
|---|---|
| `outputs/validation_predictions.jsonl` | Final prediction records |
| `outputs/validation_report.json` | Benchmark evaluation report |
| `outputs/artifacts/experiment_summary.json` | Training and experiment details |
| `outputs/artifacts/privacy_summary.json` | Privacy prototype details |

## 8. Testing

 The tests check:

- Personal information detection
- De-identification rendering
- Structured extraction behavior
- Submission output format
- FedAvg sample weighting
- Differential privacy loss suppression
- Differential privacy accounting
- Prediction independence from validation labels

Run the test suite with:

```powershell
python -m pytest -q tests
```

The combined project test suite contains nine tests, all of which pass.

## 9. Limitations

This solution has some limits:

- It uses a small, synthetic dataset.  
- The validation data is public.  
- The text rules may not work well with different note formats or new words.  
- The federated learning system runs as a simulation on one computer.  
- Secure aggregation is not included.  
- The final predictions do not use differential privacy.  
- The results should not be used for real medical decisions.

## 10. Conclusion

This project uses clinical text from different hospitals to:

- Remove personal information
- Extract important clinical details
- Predict whether a patient may be readmitted

It uses federated learning, so hospitals can train a shared model without sharing raw patient data. It also includes a privacy prototype using differential privacy.

The system received full scores for de-identification and data extraction, and a good score for readmission prediction on public synthetic data. More testing with real clinical data is needed before real-world use..
```