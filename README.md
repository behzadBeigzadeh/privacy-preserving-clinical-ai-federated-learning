```markdown
# Privacy-Preserving Clinical with Federated Learning

This project is a clinical AI challenge solution that performs three tasks:

1. Removes personal information from clinical notes
2. Extracts structured clinical information from free text
3. Predicts 30-day hospital readmission using federated learning

The project uses rule-based text processing for de-identification and information extraction. The readmission model is a NumPy logistic regression model trained with FedAvg across three simulated hospital sites.

## Results

The public benchmark score is:

| Task | Score |
|---|---:|
| De-identification | 15 / 15 |
| Structured extraction | 15 / 15 |
| Readmission prediction | 7.88 / 10 |
| Total | 37.88 / 40 |

These results are based on a small public synthetic validation dataset and should not be interpreted as clinical performance.

## Project Structure

```text
.
├── data/                   # Training and validation data
├── evaluator/              # Benchmark evaluation script
├── schemas/                # Input and output schemas
├── src/
│   ├── baseline.py         # Text de-identification and extraction
│   └── federated.py        # Federated learning implementation
├── tests/                  # Automated tests
├── run_submission.py       # Main training and prediction script
└── REPORT.md               # Detailed project report

```

## Requirements

- Python 3.10 or later
- pip

## Setup

Clone the repository:

```powershell
git clone https://github.com/behzadBeigzadeh/privacy-preserving-clinical-ai-federated-learning.git
cd privacy-preserving-clinical-ai-federated-learning
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

Optionally, you can create a virtual environment before installing the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Project

Run the complete submission pipeline:

```powershell
python run_submission.py `
  --train data/train.jsonl `
  --input data/validation_inputs.jsonl `
  --output outputs/validation_predictions.jsonl `
  --artifacts-dir outputs/artifacts
```


## Evaluate the Output

```powershell
python evaluator/evaluate.py `
  --inputs data/validation_inputs.jsonl `
  --ground-truth data/validation_ground_truth.jsonl `
  --predictions outputs/validation_predictions.jsonl `
  --report outputs/validation_report.json
```

## Run Tests

```powershell
python -m pytest -q tests
```

## Output Files

After running the project, the main output files are:

| File | Description |
|---|---|
| `outputs/validation_predictions.jsonl` | Final prediction file |
| `outputs/validation_report.json` | Benchmark evaluation report |
| `outputs/artifacts/experiment_summary.json` | Model training details |
| `outputs/artifacts/privacy_summary.json` | Privacy prototype details |

## Limitations

- The dataset is synthetic and small.
- The validation dataset is public.
- The text extraction rules depend on known note formats and vocabulary.
- The federated learning system is a local simulation.
- Secure aggregation is not implemented.
- The final predictions do not use differential privacy.
- The results should not be used for real medical decisions.

## Author

Behzad Beigzadeh  

```