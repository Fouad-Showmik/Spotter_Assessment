# Freight Rate Prediction Challenge

## Project Overview

This project develops a machine learning model to predict freight load rates.

The labeled development data in `data/train-test.csv` was used for exploratory data analysis, preprocessing, feature engineering, model comparison, and time-aware validation.

Three models were evaluated:

- Median baseline
- LightGBM
- CatBoost

Expanding walk-forward validation was used because the data is time ordered. CatBoost was selected based on the validation results and then retrained on all 48,000 labeled development rows for the final predictions.

## Project Structure

```text
spotter-assessment/

│

├── data/

│   ├── train-test.csv

│   ├── validation.csv

│   ├── validation-predictions-template.csv

│   └── december-chart-inputs.csv

│

├── src/

│   ├── __init__.py

│   ├── preprocessing.py

│   ├── features.py

│   ├── validation.py

│   ├── models.py

│   ├── train.py

│   └── predict.py

│

├── notebooks/

│   └── 01_eda.ipynb

│

├── outputs/

│   └── final_model.pkl

│

├── scorer_results/

│   └── candidate_december.png

│

├── train.py

├── predict.py

├── score.py

├── requirements.txt

├── validation_predictions.csv

└── README.md
```

## Setup

Create and activate a Python virtual environment, then install the required dependencies.

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

### 1. Train the final model

Run:

```bash
python train.py
```

This trains the final CatBoost model on the full labeled development dataset and saves it to:

```text
outputs/final_model.pkl
```

### 2. Generate predictions

Run:

```bash
python predict.py
```

This generates predictions for all 12,000 rows in `data/validation.csv`, fills the supplied validation prediction template, and saves the completed submission as:

```text
validation_predictions.csv
```

It also generates the predictions for the December scenario in:

```text
data/december-chart-inputs.csv
```

### 3. Run the scorer

Run:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

The scorer validates the prediction files and creates the required December chart:

```text
scorer_results/candidate_december.png
```