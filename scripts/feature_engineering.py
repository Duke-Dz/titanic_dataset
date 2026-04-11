"""Engineer Titanic features from the cleaned datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path

from data_cleaning import (
    DATA_DIR,
    PROJECT_ROOT,
    TEST_CLEANED_PATH,
    TRAIN_CLEANED_PATH,
    format_number,
    load_rows,
    main as run_cleaning,
    safe_float,
    write_rows,
)

TRAIN_FEATURES_PATH = DATA_DIR / "train_features.csv"
TEST_FEATURES_PATH = DATA_DIR / "test_features.csv"
FEATURE_REPORT_PATH = DATA_DIR / "feature_engineering_report.json"

SCALED_COLUMNS = [
    "Age",
    "Fare",
    "FamilySize",
    "FarePerPerson",
    "LogAge",
    "LogFare",
    "AgeClassInteraction",
    "PclassFareInteraction",
]
FIXED_AGE_GROUPS = ["Child", "Teen", "Adult", "Senior"]


def safe_int(value: str | float | int | None) -> int | None:
    numeric = safe_float(value)
    return None if numeric is None else int(numeric)


def repo_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def deck_from_cabin(cabin: str) -> str:
    normalized = cabin.strip()
    return normalized[0] if normalized and normalized != "Unknown" else "Unknown"


def age_group(age: float) -> str:
    if age < 13:
        return "Child"
    if age < 20:
        return "Teen"
    if age < 60:
        return "Adult"
    return "Senior"


def slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_std(values: list[float], mean_value: float) -> float:
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance) or 1.0


def ensure_cleaned_inputs() -> None:
    if not TRAIN_CLEANED_PATH.exists() or not TEST_CLEANED_PATH.exists():
        run_cleaning()


def add_base_engineered_features(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    engineered_rows: list[dict[str, str]] = []
    for row in rows:
        enriched = dict(row)
        age = safe_float(enriched["Age"]) or 0.0
        fare = safe_float(enriched["Fare"]) or 0.0
        sib_sp = safe_int(enriched["SibSp"]) or 0
        parch = safe_int(enriched["Parch"]) or 0
        pclass = safe_int(enriched["Pclass"]) or 0
        family_size = sib_sp + parch + 1

        enriched["FamilySize"] = str(family_size)
        enriched["IsAlone"] = "1" if family_size == 1 else "0"
        enriched["Deck"] = deck_from_cabin(enriched["Cabin"])
        enriched["AgeGroup"] = age_group(age)
        enriched["FarePerPerson"] = format_number(fare / family_size)
        enriched["LogFare"] = format_number(math.log1p(max(fare, 0.0)))
        enriched["LogAge"] = format_number(math.log1p(max(age, 0.0)))
        enriched["PclassFareInteraction"] = format_number(pclass * fare)
        enriched["AgeClassInteraction"] = format_number(age * pclass)
        engineered_rows.append(enriched)
    return engineered_rows


def collect_categories(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    combined_rows = train_rows + test_rows
    return {
        "Sex": sorted({row["Sex"] for row in combined_rows}),
        "Embarked": sorted({row["Embarked"] for row in combined_rows}),
        "CleanTitle": sorted({row["CleanTitle"] for row in combined_rows}),
        "Deck": sorted({row["Deck"] for row in combined_rows}),
        "AgeGroup": FIXED_AGE_GROUPS,
    }


def build_scaling_statistics(train_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    statistics: dict[str, dict[str, float]] = {}
    for column in SCALED_COLUMNS:
        values = [safe_float(row[column]) or 0.0 for row in train_rows]
        mean_value = compute_mean(values)
        std_value = compute_std(values, mean_value)
        statistics[column] = {
            "mean": mean_value,
            "std": std_value,
        }
    return statistics


def append_encoded_and_scaled_features(
    rows: list[dict[str, str]],
    categories: dict[str, list[str]],
    scaling_statistics: dict[str, dict[str, float]],
) -> list[dict[str, str]]:
    final_rows: list[dict[str, str]] = []
    for row in rows:
        encoded = dict(row)

        for column, options in categories.items():
            for option in options:
                encoded[f"{column}_{slugify(option)}"] = "1" if row[column] == option else "0"

        for column, stats in scaling_statistics.items():
            value = safe_float(row[column]) or 0.0
            scaled = (value - stats["mean"]) / stats["std"]
            encoded[f"{column}Scaled"] = format_number(scaled)

        final_rows.append(encoded)

    return final_rows


def build_report(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    categories: dict[str, list[str]],
    scaling_statistics: dict[str, dict[str, float]],
) -> dict[str, object]:
    return {
        "row_counts": {
            "train": len(train_rows),
            "test": len(test_rows),
        },
        "engineered_features": [
            "FamilySize",
            "IsAlone",
            "Deck",
            "AgeGroup",
            "FarePerPerson",
            "LogFare",
            "LogAge",
            "PclassFareInteraction",
            "AgeClassInteraction",
            "One-hot encoded Sex, Embarked, CleanTitle, Deck, and AgeGroup",
            "Scaled numeric columns for model-ready features",
        ],
        "categories": categories,
        "scaling_statistics": {
            column: {
                "mean": format_number(stats["mean"]),
                "std": format_number(stats["std"]),
            }
            for column, stats in scaling_statistics.items()
        },
        "output_files": {
            "train_features": repo_relative(TRAIN_FEATURES_PATH),
            "test_features": repo_relative(TEST_FEATURES_PATH),
        },
    }


def main() -> None:
    ensure_cleaned_inputs()
    cleaned_train_rows = load_rows(TRAIN_CLEANED_PATH)
    cleaned_test_rows = load_rows(TEST_CLEANED_PATH)

    train_rows = add_base_engineered_features(cleaned_train_rows)
    test_rows = add_base_engineered_features(cleaned_test_rows)

    categories = collect_categories(train_rows, test_rows)
    scaling_statistics = build_scaling_statistics(train_rows)

    final_train_rows = append_encoded_and_scaled_features(train_rows, categories, scaling_statistics)
    final_test_rows = append_encoded_and_scaled_features(test_rows, categories, scaling_statistics)

    write_rows(TRAIN_FEATURES_PATH, final_train_rows)
    write_rows(TEST_FEATURES_PATH, final_test_rows)

    report = build_report(final_train_rows, final_test_rows, categories, scaling_statistics)
    FEATURE_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {TRAIN_FEATURES_PATH.name}, {TEST_FEATURES_PATH.name}, and {FEATURE_REPORT_PATH.name}")


if __name__ == "__main__":
    main()
