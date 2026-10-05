# Challenge report

## 1. Executive summary

This submission combines the Candidate prediction code with the reporting and test coverage prepared in the Amir project. Rule based processing removes the benchmark's personal identifiers and extracts the required clinical fields. A NumPy logistic model is trained with FedAvg across three hospital partitions; predictions average three fixed initialization seeds. The public evaluator records **37.88/40**: 15/15 for de-identification, 15/15 for extraction, and 7.88/10 for readmission prediction. The 30-case validation set has six positive outcomes. These are results on a small synthetic benchmark, not evidence of clinical performance. Standard predictions use non-private FedAvg. A separate client-level DP prototype is reported for the privacy exercise.

## 2. System architecture

`run_submission.py` reads JSONL inputs, runs the text and risk components, and writes predictions and experiment artifacts. `src/baseline.py` implements PII detection, placeholder rendering, and clinical extraction. `src/federated.py` implements feature construction, logistic training, FedAvg, centralized and local references, metrics, and the DP prototype. The supplied `evaluator/evaluate.py` scores the final JSONL output.

Training rows are partitioned by hospital before FedAvg. Clients contribute model updates weighted by their local sample counts. The coordinator simulates all clients in one process; it does not provide network isolation or secure aggregation.

## 3. De-identification

The detector uses field cues and note formats to locate the eight required entity types. It returns offsets in the original note, sorts the spans, and renders placeholders while retaining text between spans. The detector is rule based and tuned to the supplied synthetic note layouts.

On the 30 public validation cases, entity detection, character-level detection, and rendered de-identified text each score 1.0. This is a development result because the public note formats are available. New names without context cues, unseen address layouts, OCR errors, and other document formats may be missed or over-redacted. An image workflow would need to map detected text offsets back to image regions and verify that the visual marks cover the source text.

## 4. Structured extraction and standardization

The extractor normalizes the benchmark's diagnosis and medication vocabulary, checks nearby context for negation and status, parses documented numeric fields, converts supported units, and returns null for missing values. Smoking and allergy values use the required categories. The public validation extraction score is 1.0. Since public examples informed the rules, this should not be treated as an independent estimate of hidden-set performance. The approach remains limited to its rules and vocabulary.

## 5. Federated-learning experiment

The model uses structured features and hospital identity: age, prior admissions, length of stay, emergency admission, sex indicators, and three hospital indicators. Numeric scales are fixed in code. Logistic training uses batch gradient updates, learning rate 1.0, and L2 penalty 0.001 on non-intercept coefficients.

The training file contains 120 records across Berlin (42), Chennai (39), and Hyderabad (39). Each hospital performs one local gradient step per round; the default run uses 2,000 FedAvg rounds. Client updates are averaged in proportion to each hospital's record count. Centralized and per-site local models are computed as references. The final readmission probabilities average the three fixed FedAvg seeds 7, 19, and 43; the validation labels do not participate in training or prediction.

The public evaluator reports readmission score 0.7876, ROC-AUC 0.7986, average precision 0.7107, Brier score 0.1467, and log loss 0.4714. Per-site ROC-AUC is 0.9048 for Berlin, 1.0000 for Chennai, and 0.2222 for Hyderabad. Each site has only ten validation examples, and Hyderabad has one positive case, so those site metrics are highly unstable.

## 6. Privacy extension and threat model

The separate prototype protects one hospital update under a trusted-server assumption. It clips each full client update to norm 0.5, weights updates using the fixed public site sizes, and adds fresh Gaussian noise. With noise multiplier 4.0 over 20 rounds and delta 1e-5, the zCDP conversion reports epsilon approximately 5.9899 for client-level replacement under fixed client counts.

This prototype is separate from the standard prediction path. Standard predictions and ordinary experiment diagnostics are not differentially private. The server sees each client update; there is no secure aggregation. The DP run does not calculate or export raw client training losses. The guarantee does not cover changing hospital counts, patient-level replacement, or other releases. It is a mathematical prototype, not a production privacy implementation.

## 7. Reproducibility and testing

From the repository root, `python run_submission.py --train data/train.jsonl --input data/validation_inputs.jsonl --output outputs/validation_predictions.jsonl --artifacts-dir outputs/artifacts` creates predictions and the experiment and privacy summaries. Run `python evaluator/evaluate.py --inputs data/validation_inputs.jsonl --ground-truth data/validation_ground_truth.jsonl --predictions outputs/validation_predictions.jsonl --report outputs/validation_report.json` to score those predictions. Ground truth is read only by the evaluator in this workflow.

The tests cover the supplied PII spans in original and flattened notes, the evaluator contract, FedAvg sample weighting, DP loss suppression and accounting, and prediction independence from evaluation labels. All 9 tests pass in the combined repository. Normal training uses fixed seeds. DP noise is fresh and therefore the DP artifact can vary across runs. The public benchmark run scored 37.8757/40 after integration; the risk-model algorithm is unchanged from Candidate.

## 8. Limitations and next steps

The dataset is synthetic and small, the validation set is public, and rule vocabulary was informed by public examples. Results may not generalize to unseen language or real clinical notes. The federated code is a single-process simulation, and the privacy path has a trusted coordinator and no secure aggregation. Hidden evaluation and a Docker image build have not been run for this combined folder.
