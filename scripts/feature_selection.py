"""Select model-ready Titanic features using correlation and information gain."""

from __future__ import annotations

import json
import math
from pathlib import Path

from data_cleaning import DATA_DIR, PROJECT_ROOT, format_number, load_rows, safe_float, write_rows
from feature_engineering import TEST_FEATURES_PATH, TRAIN_FEATURES_PATH, main as run_feature_engineering

TRAIN_SELECTED_PATH = DATA_DIR / "train_selected.csv"
TEST_SELECTED_PATH = DATA_DIR / "test_selected.csv"
FEATURE_SELECTION_REPORT_PATH = DATA_DIR / "feature_selection_report.json"

EXCLUDED_COLUMNS = {
    "PassengerId",
    "Survived",
    "Name",
    "Ticket",
    "Cabin",
    "Sex",
    "Embarked",
    "CleanTitle",
    "Deck",
    "AgeGroup",
}
CORRELATION_THRESHOLD = 0.85
MAX_SELECTED_FEATURES = 12


def repo_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def ensure_feature_inputs() -> None:
    if not TRAIN_FEATURES_PATH.exists() or not TEST_FEATURES_PATH.exists():
        run_feature_engineering()


def entropy(labels: list[int]) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    value = 0.0
    for count in counts.values():
        probability = count / total
        value -= probability * math.log2(probability)
    return value


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_denominator = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_denominator == 0 or right_denominator == 0:
        return 0.0
    return numerator / (left_denominator * right_denominator)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    midpoint = count // 2
    if count % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def information_gain(values: list[float], labels: list[int]) -> tuple[float, float | None]:
    unique_values = sorted(set(values))
    if len(unique_values) <= 1:
        return 0.0, None

    if len(unique_values) == 2:
        threshold = sum(unique_values) / 2
    else:
        threshold = median(values)

    left_labels = [label for value, label in zip(values, labels) if value <= threshold]
    right_labels = [label for value, label in zip(values, labels) if value > threshold]
    base_entropy = entropy(labels)
    weighted_entropy = (
        (len(left_labels) / len(labels)) * entropy(left_labels)
        + (len(right_labels) / len(labels)) * entropy(right_labels)
    )
    return base_entropy - weighted_entropy, threshold


def numeric_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    for column in rows[0]:
        if column in EXCLUDED_COLUMNS:
            continue
        values = [safe_float(row[column]) for row in rows]
        if all(value is not None for value in values):
            columns.append(column)
    return columns


def column_values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [safe_float(row[column]) or 0.0 for row in rows]


def feature_ranking(rows: list[dict[str, str]], columns: list[str]) -> list[dict[str, object]]:
    target = [int(float(row["Survived"])) for row in rows]
    raw_rankings: list[dict[str, object]] = []

    for column in columns:
        values = column_values(rows, column)
        correlation = pearson_correlation(values, target)
        gain, threshold = information_gain(values, target)
        raw_rankings.append(
            {
                "feature": column,
                "target_correlation": correlation,
                "information_gain": gain,
                "threshold": threshold,
            }
        )

    max_correlation = max((abs(item["target_correlation"]) for item in raw_rankings), default=1.0) or 1.0
    max_gain = max((item["information_gain"] for item in raw_rankings), default=1.0) or 1.0

    for item in raw_rankings:
        correlation_score = abs(item["target_correlation"]) / max_correlation
        gain_score = item["information_gain"] / max_gain
        item["score"] = 0.55 * correlation_score + 0.45 * gain_score

    return sorted(
        raw_rankings,
        key=lambda item: (item["score"], abs(item["target_correlation"]), item["information_gain"]),
        reverse=True,
    )


def select_features(rows: list[dict[str, str]], rankings: list[dict[str, object]]) -> tuple[list[str], list[dict[str, object]]]:
    selected: list[str] = []
    dropped: list[dict[str, object]] = []

    for item in rankings:
        feature = str(item["feature"])
        current_values = column_values(rows, feature)
        conflict_feature = None
        conflict_correlation = 0.0

        for chosen in selected:
            redundancy = abs(pearson_correlation(current_values, column_values(rows, chosen)))
            if redundancy >= CORRELATION_THRESHOLD:
                conflict_feature = chosen
                conflict_correlation = redundancy
                break

        if conflict_feature is not None:
            dropped.append(
                {
                    "feature": feature,
                    "dropped_because": conflict_feature,
                    "correlation": format_number(conflict_correlation),
                }
            )
            continue

        selected.append(feature)
        if len(selected) >= MAX_SELECTED_FEATURES:
            break

    return selected, dropped


def subset_rows(rows: list[dict[str, str]], selected_columns: list[str], include_target: bool) -> list[dict[str, str]]:
    subset: list[dict[str, str]] = []
    for row in rows:
        filtered = {"PassengerId": row["PassengerId"]}
        if include_target:
            filtered["Survived"] = row["Survived"]
        for column in selected_columns:
            filtered[column] = row[column]
        subset.append(filtered)
    return subset


def build_report(
    columns: list[str],
    rankings: list[dict[str, object]],
    selected: list[str],
    dropped: list[dict[str, object]],
) -> dict[str, object]:
    formatted_rankings: list[dict[str, object]] = []
    for item in rankings:
        formatted_rankings.append(
            {
                "feature": item["feature"],
                "score": format_number(item["score"]),
                "target_correlation": format_number(item["target_correlation"]),
                "information_gain": format_number(item["information_gain"]),
                "threshold": None if item["threshold"] is None else format_number(item["threshold"]),
            }
        )

    return {
        "candidate_feature_count": len(columns),
        "selection_rules": {
            "correlation_threshold": CORRELATION_THRESHOLD,
            "max_selected_features": MAX_SELECTED_FEATURES,
            "importance_formula": "0.55 * normalized absolute target correlation + 0.45 * normalized information gain",
        },
        "selected_features": selected,
        "dropped_as_redundant": dropped,
        "ranking": formatted_rankings,
        "output_files": {
            "train_selected": repo_relative(TRAIN_SELECTED_PATH),
            "test_selected": repo_relative(TEST_SELECTED_PATH),
        },
    }


def main() -> None:
    ensure_feature_inputs()
    train_rows = load_rows(TRAIN_FEATURES_PATH)
    test_rows = load_rows(TEST_FEATURES_PATH)

    columns = numeric_feature_columns(train_rows)
    rankings = feature_ranking(train_rows, columns)
    selected, dropped = select_features(train_rows, rankings)

    train_subset = subset_rows(train_rows, selected, include_target=True)
    test_subset = subset_rows(test_rows, selected, include_target=False)

    write_rows(TRAIN_SELECTED_PATH, train_subset)
    write_rows(TEST_SELECTED_PATH, test_subset)

    report = build_report(columns, rankings, selected, dropped)
    FEATURE_SELECTION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {TRAIN_SELECTED_PATH.name}, {TEST_SELECTED_PATH.name}, and {FEATURE_SELECTION_REPORT_PATH.name}")


if __name__ == "__main__":
    main()
