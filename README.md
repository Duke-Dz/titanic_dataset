# Titanic Survival Assignment

This repository contains the full solution for the Titanic dataset assignment. It covers data cleaning, feature engineering, feature selection.

## Project structure

```text
titanic_assignment/
|-- data/
|   |-- train.csv
|   |-- test.csv
|   |-- train_cleaned.csv
|   |-- test_cleaned.csv
|   |-- train_features.csv
|   |-- test_features.csv
|   |-- train_selected.csv
|   |-- test_selected.csv
|   |-- cleaning_report.json
|   |-- feature_engineering_report.json
|   `-- feature_selection_report.json
|-- notebooks/
|   `-- Titanic_Feature_Engineering.ipynb
|-- scripts/
|   |-- data_cleaning.py
|   |-- feature_engineering.py
|   `-- feature_selection.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Assignment coverage

### Part 1: Data cleaning
- Identified missing values in the train and test sets.
- Imputed `Age` using median age by `CleanTitle` and `Pclass`, with fallback to title-level and global medians.
- Imputed missing `Fare` using the training median of the matching `Pclass`.
- Imputed missing `Embarked` with the training-set mode, `S`.
- Preserved `Cabin` for deck extraction, replaced blanks with `Unknown`, and created missingness indicators.
- Standardized `Sex` values to lowercase.
- Checked duplicates and handled them in the cleaning pipeline if present.
- Capped `Age` and `Fare` outliers using IQR bounds from the cleaned training set.

### Part 2: Feature engineering
- Created `FamilySize`, `IsAlone`, `Deck`, `AgeGroup`, `FarePerPerson`, `LogAge`, `LogFare`, `AgeClassInteraction`, and `PclassFareInteraction`.
- One-hot encoded `Sex`, `Embarked`, `CleanTitle`, `Deck`, and `AgeGroup`.
- Added scaled versions of key numeric features for model-ready data.

### Part 3: Feature selection
- Ranked numeric features using a combined score based on absolute target correlation and information gain.
- Removed redundant features using a pairwise correlation threshold of `0.85`.
- Produced final reduced datasets for training and test use.

## Data cleaning summary

### Missing values before cleaning
- Training set: `Age=177`, `Cabin=687`, `Embarked=2`
- Test set: `Age=86`, `Cabin=327`, `Fare=1`

### Reference statistics from training data
- `Embarked` mode: `S`
- Fare median by class:
  `Pclass 1 = 60.2875`, `Pclass 2 = 14.25`, `Pclass 3 = 8.05`
- Global age median fallback: `28`

### Outlier caps
- `Age`: lower `-2.625`, upper `60.375`
- `Fare`: lower `-26.724`, upper `65.6344`

## Selected features

The final selected features are:
- `CabinWasMissing`
- `LogFare`
- `FarePerPerson`
- `Pclass`
- `PclassFareInteraction`
- `IsAlone`
- `LogAge`
- `Parch`
- `FamilySize`
- `AgeWasMissing`
- `EmbarkedWasMissing`
- `FareWasMissing`

## Key findings

- Missing cabin information is itself informative, which is why `CabinWasMissing` remains in the selected set.
- Fare-based transformations add value, but raw `Fare` was dropped because `LogFare` captured similar signal with less redundancy.
- Family context remains useful in the final dataset through `FamilySize`, `Parch`, and `IsAlone`.
- The processed train and test outputs contain no missing values.

## How to run

Run the pipeline from the project root:

```powershell
python scripts\data_cleaning.py
python scripts\feature_engineering.py
python scripts\feature_selection.py
```

## Main outputs

Generated files in `data/`:
- `train_cleaned.csv` and `test_cleaned.csv`
- `train_features.csv` and `test_features.csv`
- `train_selected.csv` and `test_selected.csv`
- `cleaning_report.json`
- `feature_engineering_report.json`
- `feature_selection_report.json`

## Notebook

`notebooks/Titanic_Feature_Engineering.ipynb` contains a submission-ready walkthrough of the pipeline with cleaning decisions, engineered-feature previews, transformation plots, and the final selected feature set. It can be opened directly in Jupyter or previewed on GitHub.

## Requirements

The scripts run with the Python standard library. `requirements.txt` includes only optional notebook and visualization tools.
