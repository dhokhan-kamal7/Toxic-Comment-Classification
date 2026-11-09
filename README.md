# Toxic Comment Classification: Transfer Learning, Fine-Tuning, Hyperparameter Tuning, and Interpretability

## Project Overview

This project classifies user comments as **toxic** or **non-toxic** using a BERT-based language model.

The pipeline covers four major stages:

1. **Transfer Learning:** Uses a frozen pre-trained BERT model as a feature extractor to establish a baseline.
2. **Fine-Tuning:** Fine-tunes the pre-trained BERT model end-to-end for toxic comment classification.
3. **Hyperparameter Tuning:** Uses Optuna to optimize the fine-tuned model's hyperparameters.
4. **Interpretability:** Applies SHAP to explain selected model predictions.

The trained models are also evaluated on real-world-style text examples.

## Installation

Install the required dependencies using:

```bash
pip install transformers datasets optuna shap accelerate evaluate scikit-learn matplotlib seaborn pandas
```

## Dataset

- **Dataset:** Civil Comments (`google/civil_comments`)
- **Loading:** `load_dataset("google/civil_comments")`
- **Text Column:** `text`
- **Score Column:** `toxicity` (ranges from 0 to 1).

### Label Generation

Comments are converted into binary labels using the following threshold:

- `toxicity >= 0.5` → **Toxic (1)**
- `toxicity < 0.5` → **Non-toxic (0)**

### Preprocessing and Data Splits

The following preprocessing steps are applied:

- Duplicate comments are removed within each official dataset split.
- Training comments that also appear in the official validation or test split are removed to prevent data leakage.
- The training set is constructed using a balanced random subset of the official training split.
- Validation and test sets retain their natural class distributions.

| Dataset Split | Sample Size | Class Distribution |
|---|---:|---|
| Training | 20,000 | 10,000 toxic + 10,000 non-toxic |
| Validation | 5,000 | Natural distribution |
| Test | 10,000 | Natural distribution |

**Additional preprocessing details:**

- Maximum sequence length: 128 tokens.
- Tokenization is performed once per split before building the DataLoaders.
- The training set is balanced, while validation and test sets preserve their original imbalanced distributions.

Consequently, F1-score, precision, recall, and ROC-AUC are particularly informative evaluation metrics alongside accuracy.

## Model Development

### 1. Frozen Baseline (Transfer Learning)

The pre-trained BERT model is used as a feature extractor.

**Configuration:**

- All BERT parameters are frozen.
- BERT remains in evaluation mode during training to disable dropout in the feature extractor.
- The token representation is passed to a trainable classification head consisting of:
  - Dropout (0.1)
  - Linear classification layer
- **Optimizer:** AdamW
- **Learning Rate:** 1e-3
- **Epochs:** 5
- **Loss Function:** Cross-Entropy Loss

### 2. Fine-Tuning

The pre-trained model is fine-tuned end-to-end using `AutoModelForSequenceClassification`.

**Default Configuration:**

| Hyperparameter | Value |
|---|---|
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Epochs | 3 |
| Optimizer | AdamW |
| Scheduler | Linear Warmup/Decay |

**Additional training techniques:**

- All model parameters are trainable.
- Gradient clipping (maximum norm: 1.0).
- Mixed-precision training on GPU.
- The best validation-F1 epoch weights are restored at the end of training.

### 3. Hyperparameter Tuning with Optuna

Optuna is used to identify better hyperparameter configurations for the fine-tuned model.

**Configuration:**

- **Optimization Method:** Optuna with TPE sampler.
- **Number of Trials:** 12.
- **Objective:** Maximize validation F1-score.
- **Test Set:** Not used during hyperparameter tuning.

**Search Space:**

| Hyperparameter | Search Space |
|---|---|
| Learning Rate | 1e-5 to 5e-5 (log scale) |
| Batch Size | 16, 32, 64 |
| Weight Decay | 0.0 to 0.1 |
| Warmup Ratio | 0.0 to 0.2 |
| Epochs | 2 to 3 |

The best hyperparameter configuration is selected based on validation F1-score.

### 4. Interpretability with SHAP

SHAP (SHapley Additive exPlanations) is used to interpret selected model predictions.

It helps identify which words or text segments contribute to the model's toxicity predictions, providing insights into how the classifier makes its decisions.

## Evaluation Protocol

The evaluation process is designed to prevent test-set leakage.

- **Validation Set:** Used for model development and comparison between the frozen baseline and default fine-tuning.
- **Test Set:** Used once at the end to evaluate all three models together:
  1. Frozen BERT baseline.
  2. Fine-tuned BERT with default hyperparameters.
  3. Fine-tuned BERT with Optuna-tuned hyperparameters.

### Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## How to Run

1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime.
3. Run all notebook cells sequentially.
4. The first cell installs the required dependencies.
5. Follow the notebook to execute preprocessing, model training, hyperparameter tuning, evaluation, and SHAP interpretation.

