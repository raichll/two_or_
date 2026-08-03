"""Build the audited staffing evidence used by the IISE submission.

The script merges two common-parameter routing grids, keeps the twelve held-out
dates used in the paper, and selects a workforce size only after verifying that
each date contains one candidate solution for every size in {1, ..., 5}.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, stdev


PAPER_ROOT = Path(__file__).resolve().parents[1]
PAPER_OUTPUTS = PAPER_ROOT / "outputs"

DATES = (
    "2024-12-02",
    "2024-12-03",
    "2024-12-04",
    "2024-12-05",
    "2024-12-06",
    "2024-12-15",
    "2024-12-16",
    "2024-12-17",
    "2024-12-23",
    "2024-12-24",
    "2024-12-26",
    "2024-12-31",
)
TEAM_COUNTS = (1, 2, 3, 4, 5)
TEAM_COSTS = (0.50, 1.00, 1.50, 2.00)
METRICS = (
    "worst_case_objective",
    "nominal_objective",
    "realized_recourse_value",
    "positives_found",
    "positive_coverage",
    "distance_km",
    "expected_unresolved_count",
    "runtime_seconds",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    observed = [value for value in values if math.isfinite(value)]
    if not observed:
        return math.nan, math.nan
    return fmean(observed), stdev(observed) if len(observed) > 1 else 0.0


def to_float(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help=(
            "One or more common-parameter staffing grids. Defaults to the "
            "public 60-row canonical grid in outputs/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_OUTPUTS,
        help="Directory for the canonical grid and summary CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = args.input or [PAPER_OUTPUTS / "iise_staffing_grid_canonical.csv"]
    raw: list[dict[str, str]] = []
    for input_path in input_paths:
        raw.extend(read_rows(input_path))
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for row in raw:
        date = row["date"]
        teams = int(row["vehicles"])
        if date not in DATES or teams not in TEAM_COUNTS:
            continue
        key = (date, teams)
        if key in indexed:
            raise ValueError(f"Duplicate staffing candidate: {key}")
        converted: dict[str, object] = dict(row)
        converted["vehicles"] = teams
        for metric in METRICS:
            converted[metric] = to_float(row[metric])
        indexed[key] = converted

    expected = {(date, teams) for date in DATES for teams in TEAM_COUNTS}
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"Missing staffing candidates: {missing}")

    canonical = [indexed[(date, teams)] for date in DATES for teams in TEAM_COUNTS]
    output_dir = args.output_dir
    write_rows(output_dir / "iise_staffing_grid_canonical.csv", canonical)

    selections: list[dict[str, object]] = []
    for team_cost in TEAM_COSTS:
        for date in DATES:
            candidates = [indexed[(date, teams)] for teams in TEAM_COUNTS]
            policies = {
                "robust_endogenous": max(
                    candidates,
                    key=lambda row: float(row["worst_case_objective"])
                    - team_cost * int(row["vehicles"]),
                ),
                "nominal_value": max(
                    candidates,
                    key=lambda row: float(row["nominal_objective"])
                    - team_cost * int(row["vehicles"]),
                ),
                "fixed_two_teams": indexed[(date, 2)],
            }
            for policy, chosen in policies.items():
                teams = int(chosen["vehicles"])
                selections.append(
                    {
                        "date": date,
                        "team_cost": team_cost,
                        "policy": policy,
                        "selected_teams": teams,
                        "net_robust_value": float(chosen["worst_case_objective"])
                        - team_cost * teams,
                        "net_nominal_value": float(chosen["nominal_objective"])
                        - team_cost * teams,
                        **{metric: chosen[metric] for metric in METRICS},
                    }
                )
    write_rows(output_dir / "iise_staffing_policy_by_date.csv", selections)

    grouped: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in selections:
        grouped[(float(row["team_cost"]), str(row["policy"]))].append(row)

    summaries: list[dict[str, object]] = []
    for (team_cost, policy), rows in sorted(grouped.items()):
        team_values = [float(row["selected_teams"]) for row in rows]
        counts = Counter(int(value) for value in team_values)
        summary: dict[str, object] = {
            "team_cost": team_cost,
            "policy": policy,
            "dates": len(rows),
            "mean_selected_teams": round(fmean(team_values), 4),
            "team_selection_frequency": ";".join(
                f"{teams}:{counts.get(teams, 0)}" for teams in TEAM_COUNTS
            ),
        }
        for metric in (
            "net_robust_value",
            "net_nominal_value",
            "realized_recourse_value",
            "positives_found",
            "positive_coverage",
            "distance_km",
            "expected_unresolved_count",
        ):
            values = [float(row[metric]) for row in rows]
            avg, sd = mean_sd(values)
            summary[f"mean_{metric}"] = round(avg, 6)
            summary[f"sd_{metric}"] = round(sd, 6)
        summaries.append(summary)
    write_rows(output_dir / "iise_staffing_policy_summary.csv", summaries)

    # Paired differences isolate the staffing criterion while holding the route
    # candidate grid fixed. Positive values favor robust endogenous staffing.
    paired: list[dict[str, object]] = []
    by_key = {
        (float(row["team_cost"]), str(row["date"]), str(row["policy"])): row
        for row in selections
    }
    for team_cost in TEAM_COSTS:
        for comparator in ("nominal_value", "fixed_two_teams"):
            for metric in (
                "net_robust_value",
                "realized_recourse_value",
                "positives_found",
                "distance_km",
                "expected_unresolved_count",
            ):
                diffs = []
                for date in DATES:
                    robust = by_key[(team_cost, date, "robust_endogenous")]
                    base = by_key[(team_cost, date, comparator)]
                    sign = -1.0 if metric in {"distance_km", "expected_unresolved_count"} else 1.0
                    difference = sign * (float(robust[metric]) - float(base[metric]))
                    if math.isfinite(difference):
                        diffs.append(difference)
                avg, sd = mean_sd(diffs)
                se = sd / math.sqrt(len(diffs))
                paired.append(
                    {
                        "team_cost": team_cost,
                        "comparator": comparator,
                        "metric": metric,
                        "mean_favorable_difference": round(avg, 6),
                        "normal_95ci_low": round(avg - 1.96 * se, 6),
                        "normal_95ci_high": round(avg + 1.96 * se, 6),
                    }
                )
    write_rows(output_dir / "iise_staffing_paired_differences.csv", paired)


if __name__ == "__main__":
    main()
