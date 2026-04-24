# COMP 334 – Assignment 2: Titanic Dataset Analysis

## Overview

For this assignment, I built a complete data-processing pipeline on the **Titanic – Machine Learning from Disaster** dataset from Kaggle. The pipeline covers **data cleaning**, **feature engineering**, and **feature selection** — the three main parts the assignment requires. I used **Anaconda** and **Jupyter Notebook** for exploration and then modularised the logic into standalone Python scripts, exactly as the assignment brief suggests.

Everything is documented in both the notebook and this README so the approach, decisions, and key findings are clear.

## Project Structure

```text
titanic_assignment/
├── data/
│   ├── train.csv                      # Original training set (includes Survived)
│   ├── test.csv                       # Original test set
│   ├── train_cleaned.csv              # Output of Part 1 – cleaned training data
│   ├── test_cleaned.csv               # Output of Part 1 – cleaned test data
│   ├── train_features.csv             # Output of Part 2 – engineered features (train)
│   ├── test_features.csv              # Output of Part 2 – engineered features (test)
│   ├── train_selected.csv             # Output of Part 3 – selected features (train)
│   ├── test_selected.csv              # Output of Part 3 – selected features (test)
│   ├── cleaning_report.json           # Machine-readable cleaning report
│   ├── feature_engineering_report.json # Machine-readable engineering report
│   └── feature_selection_report.json  # Machine-readable selection report
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb  # Full walkthrough notebook
├── scripts/
│   ├── data_cleaning.py               # Part 1 script
│   ├── feature_engineering.py         # Part 2 script
│   └── feature_selection.py           # Part 3 script
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Part 1: Data Cleaning (10 Marks)

### 1. Missing Value Handling

I started by identifying every missing value in both the training and test sets:

| Column | Train Missing | Test Missing |
|--------|--------------|-------------|
| Age | 177 | 86 |
| Cabin | 687 | 327 |
| Embarked | 2 | 0 |
| Fare | 0 | 1 |

My imputation strategy for each:

- **`Age`** – I imputed missing ages using the **median age** grouped by `CleanTitle` (extracted from the Name field) and `Pclass`. If a particular title+class combination wasn't available, I fell back to the title-level median, and finally to the global median of **28**. This approach is more accurate than a simple global median because passenger titles and class strongly correlate with age.

- **`Fare`** – The single missing fare in the test set was imputed using the **training-set median fare** for the matching `Pclass`:
  - Pclass 1: **60.2875**
  - Pclass 2: **14.25**
  - Pclass 3: **8.05**

- **`Embarked`** – The two missing values were filled with the **training-set mode**, which is **S** (Southampton).

- **`Cabin`** – I kept the Cabin column for deck extraction. Missing values were replaced with `"Unknown"` and I created a `CabinWasMissing` indicator column because missingness itself turned out to be informative (it ended up as the top-ranked selected feature).

- I also created `AgeWasMissing`, `FareWasMissing`, and `EmbarkedWasMissing` indicator columns so the model can learn from the pattern of missing data.

### 2. Outlier Handling

I detected outliers in `Age` and `Fare` using the **IQR method** on the cleaned training set and then capped values at the computed bounds:

| Feature | Lower Bound | Upper Bound |
|---------|-------------|-------------|
| Age | −2.625 | 60.375 |
| Fare | −26.724 | 65.6344 |

Any value below the lower bound was capped upward; any value above the upper bound was capped downward. This avoids distortion from extreme outliers while keeping reasonable variation.

### 3. Data Consistency

- I **standardised `Sex`** values to lowercase (`male` / `female`).
- I **standardised `Embarked`** to a single uppercase letter (`S`, `C`, `Q`).
- I checked for **duplicate rows** and handled them in the pipeline (none were found in the provided data).

### 4. Deliverable

The cleaned datasets are saved as `data/train_cleaned.csv` and `data/test_cleaned.csv`. A detailed `cleaning_report.json` is also generated. Every decision is explained in the notebook with markdown cells.

---

## Part 2: Feature Engineering (30 Marks)

### 1. Derived Features I Created

| Feature | Formula / Logic |
|---------|----------------|
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `1` if `FamilySize == 1`, else `0` |
| `CleanTitle` | Extracted from `Name` (Mr, Mrs, Miss, Master, Dr, Rev, Officer, Royalty, Rare) |
| `Deck` | First letter of `Cabin` (A–G, T, or Unknown) |
| `AgeGroup` | Child (< 13), Teen (13–19), Adult (20–59), Senior (60+) |
| `FarePerPerson` | `Fare / FamilySize` |

### 2. Categorical Encoding

I **one-hot encoded** the following nominal features:
- `Sex` → `Sex_female`, `Sex_male`
- `Embarked` → `Embarked_C`, `Embarked_Q`, `Embarked_S`
- `CleanTitle` → `CleanTitle_master`, `CleanTitle_miss`, `CleanTitle_mr`, `CleanTitle_mrs`, etc.
- `Deck` → `Deck_a`, `Deck_b`, … , `Deck_unknown`
- `AgeGroup` → `AgeGroup_child`, `AgeGroup_teen`, `AgeGroup_adult`, `AgeGroup_senior`

`Pclass` was kept as a numeric ordinal feature.

### 3. Interaction Features

I created two interaction terms to capture combined effects:
- `PclassFareInteraction` = `Pclass × Fare`
- `AgeClassInteraction` = `Age × Pclass`

### 4. Feature Transformations

- **Log transforms**: I applied `log1p` to `Fare` and `Age` to produce `LogFare` and `LogAge`, reducing right skew.
- **Standardisation / Scaling**: I standardised (z-score) the following numeric columns using training-set mean and standard deviation: `Age`, `Fare`, `FamilySize`, `FarePerPerson`, `LogAge`, `LogFare`, `AgeClassInteraction`, `PclassFareInteraction`. The scaled columns are suffixed `Scaled`.

### 5. Deliverable

The engineered datasets are saved as `data/train_features.csv` and `data/test_features.csv`. The notebook shows each new feature being created and includes visualisation plots (Fare vs LogFare distributions, title counts, deck counts) to justify the transformations.

---

## Part 3: Feature Selection (10 Marks)

### 1. Correlation Analysis

I computed pairwise Pearson correlations among all numeric candidate features. Any feature that had a pairwise correlation **≥ 0.85** with an already-selected, higher-ranked feature was dropped as redundant.

### 2. Feature Importance / Ranking

I ranked features using a **combined score**:
```
score = 0.55 × normalised |target correlation| + 0.45 × normalised information gain
```

This blends linear association (correlation with `Survived`) with non-linear discriminative power (information gain based on entropy), giving a balanced ranking.

### 3. Final Selected Features

After applying the redundancy filter (threshold 0.85) and keeping the top 12 features, my final selected set is:

| # | Feature | Why Kept |
|---|---------|----------|
| 1 | `CabinWasMissing` | Missingness is itself highly predictive of survival |
| 2 | `LogFare` | Captures fare signal with reduced skew; raw `Fare` dropped as redundant |
| 3 | `FarePerPerson` | Per-person spending adds value beyond total fare |
| 4 | `Pclass` | Ticket class is a core survival predictor |
| 5 | `PclassFareInteraction` | Captures interplay between class and cost |
| 6 | `IsAlone` | Travelling alone vs with family matters |
| 7 | `LogAge` | Age signal with reduced skew |
| 8 | `Parch` | Number of parents/children aboard |
| 9 | `FamilySize` | Overall family size context |
| 10 | `AgeWasMissing` | Age missingness pattern |
| 11 | `EmbarkedWasMissing` | Embarkation missingness pattern |
| 12 | `FareWasMissing` | Fare missingness pattern |

**Dropped as redundant**: features like raw `Fare`, `SibSp`, and several one-hot columns were removed because they were highly correlated with already-selected features.

### 4. Deliverable

The reduced datasets are saved as `data/train_selected.csv` and `data/test_selected.csv`. A full ranking table with scores, correlations, and information-gain values is in `feature_selection_report.json`.

---

## Key Findings

- **Cabin missingness is the single most informative signal.** Passengers without a recorded cabin number had much lower survival rates, likely because they were in lower-class accommodation.
- **Fare-based features dominate the top rankings**, but log-transformed fare is preferred over raw fare because it reduces skew and redundancy.
- **Family context matters.** `FamilySize`, `Parch`, and `IsAlone` all survive into the final set — travelling alone reduced chances of survival.
- The final processed training and test datasets contain **zero missing values**, making them ready for any downstream model.

---

## How to Run

### Option 1 – Run the scripts from the project root

```powershell
python scripts\data_cleaning.py
python scripts\feature_engineering.py
python scripts\feature_selection.py
```

Each script automatically runs its dependencies, so you can also just run `feature_selection.py` and it will trigger cleaning and engineering if needed.

### Option 2 – Open the notebook

Open `notebooks/Titanic_Feature_Engineering.ipynb` in Jupyter Notebook or JupyterLab. It contains a full submission-ready walkthrough with cleaning decisions, engineered-feature previews, transformation plots, and the final selected feature set.

### Requirements

Install the optional notebook and visualisation tools:
```bash
pip install -r requirements.txt
```

The core scripts only use the **Python standard library** — no additional packages are needed to run them.

## Tools I Used

- **Python 3** (via Anaconda)
- **Jupyter Notebook** for interactive exploration and the submission walkthrough
- **matplotlib** for visualisation plots in the notebook
- **Python standard library** (`csv`, `json`, `math`, `re`, `collections`, `pathlib`) for the pipeline scripts
