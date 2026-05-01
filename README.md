# ML Security and Fairness Audit on COMPAS

This project analyzes how machine learning models behave under different security and fairness risks using the COMPAS recidivism dataset. The notebook trains baseline models, checks racial fairness metrics, and then tests how the models respond to adversarial attacks, data poisoning, and membership inference.

The goal is not only to measure model accuracy, but also to understand whether the model becomes unfair, unstable, or privacy-sensitive under attack.

## Project Overview

The notebook focuses on three main attack scenarios:

1. **PGD Evasion Attack**
   - Tests how small changes to input features affect model predictions.
   - Measures whether false positive rates change differently across racial groups.
   - Uses Logistic Regression as the main model for the PGD attack.

2. **Label-Flip Data Poisoning**
   - Simulates a training-time attack where some labels are intentionally changed.
   - Tests whether the model can stay accurate while fairness metrics become worse.
   - Tracks AUC, false positive rates, and Adverse Impact Ratio.

3. **Membership Inference Attack**
   - Tests whether an attacker can guess if a record was part of the training data.
   - Uses model confidence scores to compare training members and test non-members.
   - Connects privacy leakage to overfitting and generalization gap.

## Dataset

The project uses the public ProPublica COMPAS dataset:

- Dataset: COMPAS two-year recidivism data
- Source: ProPublica COMPAS Analysis
- Target variable: `two_year_recid`
- Protected/group variable used for fairness analysis: `race`

The notebook applies the same filtering conditions used in the lecture:

- Keeps records where `days_b_screening_arrest` is between -30 and 30
- Removes rows where `is_recid = -1`
- Removes rows where `c_charge_degree = O`
- Drops missing values for the selected features

## Features Used

The model uses the following features:

- `age`
- `priors_count`
- `juv_fel_count`
- `juv_misd_count`
- `juv_other_count`
- `c_charge_degree`
- `sex`

Categorical variables are one-hot encoded before training.

## Models Used

Two models are trained and compared:

- Logistic Regression
- Gradient Boosted Tree Classifier

The notebook uses a stratified train-test split and standardizes the feature values before training.

## Fairness Metrics

The main fairness metric used in the notebook is the **False Positive Rate (FPR)** by race.

The notebook also calculates **Adverse Impact Ratio (AIR)**:

```text
AIR = FPR of African-American group / FPR of Caucasian group
```

This helps show whether one group is being incorrectly flagged as high-risk at a much higher rate than another group.

## Main Results

### Clean Model Baseline

Before any attacks, the Logistic Regression model already shows a large difference in false positive rates:

- African-American FPR: 0.281
- Caucasian FPR: 0.143
- AIR: 1.961

The Gradient Boosted Tree model also shows disparity:

- African-American FPR: 0.317
- Caucasian FPR: 0.178
- AIR: 1.782

This means the fairness issue exists even before running attacks.

### PGD Evasion Attack

The PGD attack increases false positive rates for both groups as epsilon increases.

At small epsilon values, the model becomes more aggressive in predicting high risk. However, the AIR moves closer to 1.0 as epsilon becomes larger because both groups eventually receive very high false positive rates.

Key point: the attack changes model behavior strongly, even when the perturbation size is small.

### Label-Flip Poisoning

The poisoning attack flips some training labels from high-risk to low-risk for selected racial groups.

The notebook shows that AUC does not drop much, even when the fairness metric changes. This is important because accuracy alone would not reveal the full problem.

Key point: a model can still look accurate while becoming unfair.

### Membership Inference

The membership inference results are close to random guessing:

- Logistic Regression MI AUC: about 0.498
- Gradient Boosted Tree MI AUC: about 0.510

This suggests that privacy leakage is not very strong in this experiment. The larger issue in this notebook is fairness and robustness, not membership privacy.

## Important Code Note

One issue to be aware of is the PSI drift check in the poisoning section.

The code calculates PSI by comparing:

```python
psi(Xs_tr[:, j], Xs_tr[:, j])
```

Since the same data is compared to itself, PSI will always be 0. This supports the idea that label-only poisoning is hard to detect with feature drift, but a stronger version would compare clean training features with attacked or shifted feature distributions.


## Required Libraries

- pandas
- numpy
- matplotlib
- scikit-learn
- scipy


## Key Takeaway

The main takeaway from this project is that model performance is not enough. A model can have acceptable AUC and still behave unfairly across groups. Security attacks and data poisoning can make this worse, and some issues may not be caught by normal accuracy or drift monitoring.

For responsible machine learning, models should be evaluated using accuracy, fairness metrics, robustness checks, and privacy risk measures together.

