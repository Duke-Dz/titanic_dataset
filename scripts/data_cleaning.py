"""Clean the Titanic training and test datasets for the assignment."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_TRAIN_PATH = DATA_DIR / "train.csv"
RAW_TEST_PATH = DATA_DIR / "test.csv"
TRAIN_CLEANED_PATH = DATA_DIR / "train_cleaned.csv"
TEST_CLEANED_PATH = DATA_DIR / "test_cleaned.csv"
CLEANING_REPORT_PATH = DATA_DIR / "cleaning_report.json"

TITLE_PATTERN = re.compile(r",\s*([^\.]+)\.")
TITLE_NORMALIZATION = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Royalty",
    "the Countess": "Royalty",
    "Countess": "Royalty",
    "Dona": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Jonkheer": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
}
KNOWN_SEX_VALUES = {
    "male": "male",
    "m": "male",
    "female": "female",
    "f": "female",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows available to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def repo_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def safe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def format_number(value: float | int) -> str:
    numeric = float(value)
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate quantile for an empty list")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def median(values: list[float]) -> float:
    return quantile(values, 0.5)


def mode(values: list[str]) -> str:
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def extract_title(name: str) -> str:
    match = TITLE_PATTERN.search(name)
    if not match:
        return "Rare"
    raw_title = match.group(1).strip()
    return TITLE_NORMALIZATION.get(raw_title, raw_title if raw_title in {"Mr", "Mrs", "Miss", "Master", "Dr", "Rev"} else "Rare")


def standardize_sex(value: str) -> str:
    normalized = value.strip().lower()
    return KNOWN_SEX_VALUES.get(normalized, normalized or "unknown")


def standardize_embarked(value: str) -> str:
    normalized = value.strip().upper()
    return normalized[:1] if normalized else ""


def iqr_bounds(values: list[float]) -> tuple[float, float]:
    first_quartile = quantile(values, 0.25)
    third_quartile = quantile(values, 0.75)
    interquartile_range = third_quartile - first_quartile
    lower_bound = first_quartile - 1.5 * interquartile_range
    upper_bound = third_quartile + 1.5 * interquartile_range
    return lower_bound, upper_bound


def cap_value(value: float, bounds: tuple[float, float]) -> float:
    lower_bound, upper_bound = bounds
    return min(max(value, lower_bound), upper_bound)


def count_missing(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        column: sum(1 for row in rows if not str(row.get(column, "")).strip())
        for column in rows[0]
    }


def deduplicate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    deduplicated: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        marker = tuple(sorted(row.items()))
        if marker in seen:
            removed += 1
            continue
        seen.add(marker)
        deduplicated.append(row)
    return deduplicated, removed


def build_reference_statistics(train_rows: list[dict[str, str]]) -> dict[str, object]:
    embarked_mode = mode(
        [
            standardize_embarked(row["Embarked"])
            for row in train_rows
            if standardize_embarked(row["Embarked"])
        ]
    )

    fare_by_pclass: dict[str, float] = {}
    for pclass in sorted({row["Pclass"] for row in train_rows}):
        fares = [safe_float(row["Fare"]) for row in train_rows if row["Pclass"] == pclass]
        fare_values = [fare for fare in fares if fare is not None]
        fare_by_pclass[pclass] = median(fare_values)

    age_by_title_pclass: dict[tuple[str, str], float] = {}
    age_by_title: dict[str, float] = {}
    age_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    title_groups: dict[str, list[float]] = defaultdict(list)
    all_ages: list[float] = []

    for row in train_rows:
        age = safe_float(row["Age"])
        if age is None:
            continue
        title = extract_title(row["Name"])
        pclass = row["Pclass"]
        age_groups[(title, pclass)].append(age)
        title_groups[title].append(age)
        all_ages.append(age)

    for key, values in age_groups.items():
        age_by_title_pclass[key] = median(values)
    for key, values in title_groups.items():
        age_by_title[key] = median(values)

    return {
        "embarked_mode": embarked_mode,
        "fare_by_pclass": fare_by_pclass,
        "global_fare": median([safe_float(row["Fare"]) for row in train_rows if safe_float(row["Fare"]) is not None]),
        "age_by_title_pclass": age_by_title_pclass,
        "age_by_title": age_by_title,
        "global_age": median(all_ages),
    }


def impute_age(row: dict[str, str], reference_stats: dict[str, object]) -> float:
    age = safe_float(row["Age"])
    if age is not None:
        return age
    title = extract_title(row["Name"])
    pclass = row["Pclass"]
    age_by_title_pclass = reference_stats["age_by_title_pclass"]
    age_by_title = reference_stats["age_by_title"]
    if (title, pclass) in age_by_title_pclass:
        return age_by_title_pclass[(title, pclass)]
    if title in age_by_title:
        return age_by_title[title]
    return float(reference_stats["global_age"])


def impute_fare(row: dict[str, str], reference_stats: dict[str, object]) -> float:
    fare = safe_float(row["Fare"])
    if fare is not None:
        return fare
    return float(reference_stats["fare_by_pclass"].get(row["Pclass"], reference_stats["global_fare"]))


def clean_rows(
    raw_rows: list[dict[str, str]],
    reference_stats: dict[str, object],
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, str]]:
    cleaned_rows: list[dict[str, str]] = []
    for raw_row in raw_rows:
        row = dict(raw_row)

        row["Sex"] = standardize_sex(row["Sex"])

        row["EmbarkedWasMissing"] = "1" if not standardize_embarked(row["Embarked"]) else "0"
        row["Embarked"] = standardize_embarked(row["Embarked"]) or str(reference_stats["embarked_mode"])

        row["CabinWasMissing"] = "1" if not row["Cabin"].strip() else "0"
        row["Cabin"] = row["Cabin"].strip() or "Unknown"

        row["AgeWasMissing"] = "1" if safe_float(row["Age"]) is None else "0"
        age = impute_age(row, reference_stats)

        row["FareWasMissing"] = "1" if safe_float(row["Fare"]) is None else "0"
        fare = impute_fare(row, reference_stats)

        if bounds is not None:
            age = cap_value(age, bounds["Age"])
            fare = cap_value(fare, bounds["Fare"])

        row["Age"] = format_number(age)
        row["Fare"] = format_number(fare)
        row["CleanTitle"] = extract_title(row["Name"])

        cleaned_rows.append(row)

    return cleaned_rows


def duplicates_count(rows: list[dict[str, str]]) -> int:
    unique_rows = {tuple(sorted(row.items())) for row in rows}
    return len(rows) - len(unique_rows)


def build_report(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    cleaned_train_rows: list[dict[str, str]],
    cleaned_test_rows: list[dict[str, str]],
    reference_stats: dict[str, object],
    bounds: dict[str, tuple[float, float]],
    duplicates_removed: dict[str, int],
) -> dict[str, object]:
    return {
        "source_files": {
            "train": repo_relative(RAW_TRAIN_PATH),
            "test": repo_relative(RAW_TEST_PATH),
        },
        "row_counts": {
            "train": len(train_rows),
            "test": len(test_rows),
        },
        "missing_values_before": {
            "train": count_missing(train_rows),
            "test": count_missing(test_rows),
        },
        "missing_values_after": {
            "train": count_missing(cleaned_train_rows),
            "test": count_missing(cleaned_test_rows),
        },
        "duplicates_removed": {
            "train": duplicates_removed["train"],
            "test": duplicates_removed["test"],
        },
        "decisions": {
            "Age": "Median imputation using CleanTitle + Pclass groups with title and global fallbacks.",
            "Fare": "Median imputation by Pclass with a global fallback.",
            "Embarked": "Mode imputation using the training-set mode.",
            "Cabin": "Retained for deck extraction, filled with 'Unknown', and paired with CabinWasMissing indicator.",
            "Sex": "Standardized to lowercase male/female categories.",
            "Outliers": "Age and Fare capped using training-set IQR bounds.",
        },
        "reference_statistics": {
            "embarked_mode": reference_stats["embarked_mode"],
            "fare_by_pclass": reference_stats["fare_by_pclass"],
            "global_age": format_number(reference_stats["global_age"]),
            "global_fare": format_number(reference_stats["global_fare"]),
        },
        "outlier_bounds": {
            "Age": {
                "lower": format_number(bounds["Age"][0]),
                "upper": format_number(bounds["Age"][1]),
            },
            "Fare": {
                "lower": format_number(bounds["Fare"][0]),
                "upper": format_number(bounds["Fare"][1]),
            },
        },
        "output_files": {
            "train_cleaned": repo_relative(TRAIN_CLEANED_PATH),
            "test_cleaned": repo_relative(TEST_CLEANED_PATH),
        },
    }


def main() -> None:
    raw_train_rows = load_rows(RAW_TRAIN_PATH)
    raw_test_rows = load_rows(RAW_TEST_PATH)
    train_rows, train_duplicates_removed = deduplicate_rows(raw_train_rows)
    test_rows, test_duplicates_removed = deduplicate_rows(raw_test_rows)

    reference_stats = build_reference_statistics(train_rows)

    pre_capped_train_rows = clean_rows(train_rows, reference_stats)
    bounds = {
        "Age": iqr_bounds([safe_float(row["Age"]) for row in pre_capped_train_rows if safe_float(row["Age"]) is not None]),
        "Fare": iqr_bounds([safe_float(row["Fare"]) for row in pre_capped_train_rows if safe_float(row["Fare"]) is not None]),
    }

    cleaned_train_rows = clean_rows(train_rows, reference_stats, bounds)
    cleaned_test_rows = clean_rows(test_rows, reference_stats, bounds)

    write_rows(TRAIN_CLEANED_PATH, cleaned_train_rows)
    write_rows(TEST_CLEANED_PATH, cleaned_test_rows)

    report = build_report(
        train_rows=train_rows,
        test_rows=test_rows,
        cleaned_train_rows=cleaned_train_rows,
        cleaned_test_rows=cleaned_test_rows,
        reference_stats=reference_stats,
        bounds=bounds,
        duplicates_removed={
            "train": train_duplicates_removed,
            "test": test_duplicates_removed,
        },
    )
    CLEANING_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {TRAIN_CLEANED_PATH.name}, {TEST_CLEANED_PATH.name}, and {CLEANING_REPORT_PATH.name}")


if __name__ == "__main__":
    main()
