# Diabetes Regression

A beginner machine learning project that predicts diabetes disease progression
using the diabetes dataset built into scikit-learn.

The notebook `mldemo.ipynb` walks through a full regression workflow, from
exploring the data to tuning a model.

## Dataset

The [scikit-learn diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
has 442 patients and 10 features: age, sex, body mass index (`bmi`), blood
pressure (`bp`) and six blood serum measurements (`s1` to `s6`). The target is
a score measuring disease progression one year later, ranging from 25 to 346.

## Workflow

1. **Explore** the data: shape, missing values, duplicates, the target's
   distribution and a correlation heatmap.
2. **Cap outliers** in seven feature columns using the IQR rule.
3. **Split** the data into training (80%) and test (20%) sets.
4. **Standardize** the features, fitting the scaler on the training set only.
5. **Train** a linear regression model and inspect its predictions and
   coefficients.
6. **Compare** linear regression, Ridge regression and a random forest.
7. **Cross-validate** all three models with 5 folds.
8. **Tune** Ridge regression's `alpha` with a grid search.

## Results

Scores on the test set (89 patients):

| Model | R² | Mean absolute error |
|---|---|---|
| Linear Regression | 0.452 | 42.6 |
| Ridge Regression (tuned, `alpha=10`) | 0.456 | 42.8 |

Mean R² from 5-fold cross-validation on the training set:

| Model | Mean R² |
|---|---|
| Ridge Regression | 0.454 |
| Linear Regression | 0.452 |
| Random Forest | 0.393 |

## What I learned

- Linear and Ridge regression are effectively tied, and tuning `alpha` barely
  changes the score. The model explains about 45% of the variation in disease
  progression, which is about the limit for these features.
- The random forest looked almost as good on a single test split but fell
  behind in cross-validation, so one split can be misleading.
- `bmi` and `s5` have the strongest correlation with the target.
- The scaler must be fitted on the training set only, to avoid data leakage.

## How to run

Tested with Python 3.13.

```bash
git clone https://github.com/Sajoy2129/diabetes-regression.git
cd diabetes-regression

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `mldemo.ipynb` in VS Code (or Jupyter), select the `.venv` kernel and
run all cells from the top.
