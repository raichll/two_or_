import argparse
import csv
import json
import os
import time

from env_inspection_pipeline import (
    build_events,
    count_by,
    default_depot,
    evaluate,
    label_to_int,
    load_risk_file,
    parse_dt,
    read_csv_rows,
    read_json,
    run_strategy,
)


STRATEGIES = [
    "random",
    "nearest",
    "risk_first",
    "greedy_insertion",
    "alns_only",
    "hgs_only",
    "hgs_alns",
]


def select_candidate_events(events, max_events, rule):
    if len(events) <= max_events:
        return events
    if rule == "value":
        return sorted(events, key=lambda e: e["value"], reverse=True)[:max_events]
    if rule == "time":
        return sorted(events, key=lambda e: (e["event_time"], e["deadline"], e["id"]))[:max_events]
    if rule == "spatial":
        ordered = sorted(events, key=lambda e: (e["lon"], e["lat"], e["event_time"]))
        if max_events <= 1:
            return ordered[:max_events]
        step = (len(ordered) - 1) / (max_events - 1)
        chosen = []
        used = set()
        for pos in range(max_events):
            idx = round(pos * step)
            while idx in used and idx + 1 < len(ordered):
                idx += 1
            used.add(idx)
            chosen.append(ordered[idx])
        return chosen
    raise ValueError(f"Unknown selection rule: {rule}")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_dates(rows, cfg, requested=None):
    if requested:
        return requested
    date_col = cfg["date_column"]
    label_col = cfg["label_column"]
    daily = {}
    for row in rows:
        day = str(row.get(date_col, "")).strip()
        if not day:
            continue
        slot = daily.setdefault(day, {"events": 0, "positives": 0})
        slot["events"] += 1
        if label_to_int(row.get(label_col)) == 1:
            slot["positives"] += 1
    by_volume = sorted(daily, key=lambda d: daily[d]["events"], reverse=True)
    by_positive = sorted(daily, key=lambda d: daily[d]["positives"], reverse=True)
    by_date = sorted(daily)
    median_like = by_date[len(by_date) // 2 : len(by_date) // 2 + 2]
    selected = []
    for group in (by_volume[:2], by_positive[:2], median_like):
        for day in group:
            if day not in selected:
                selected.append(day)
    return selected[:6]


def metric_row(date_text, vehicle_count, strategy, metrics, runtime_seconds, event_count, risk_source):
    row = {
        "date": date_text,
        "vehicles": vehicle_count,
        "strategy": strategy,
        "event_count": event_count,
        "risk_source": risk_source,
        "objective": metrics["objective"],
        "visited_count": metrics["visited_count"],
        "positives_found": metrics["positives_found"],
        "positives_total": metrics["positives_total"],
        "positive_coverage": metrics["positive_coverage"],
        "positives_per_patrol_hour": metrics["positives_per_patrol_hour"],
        "distance_km": metrics["distance_km"],
        "lateness_hours": metrics["lateness_hours"],
        "overtime_hours": metrics["overtime_hours"],
        "vehicle_utilization": metrics["vehicle_utilization"],
        "runtime_seconds": runtime_seconds,
    }
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}


def summarize(rows):
    summary = []
    by_key = {}
    for row in rows:
        key = (row["vehicles"], row["strategy"])
        by_key.setdefault(key, []).append(row)
    metrics = [
        "objective",
        "positives_found",
        "positive_coverage",
        "positives_per_patrol_hour",
        "distance_km",
        "lateness_hours",
        "overtime_hours",
        "runtime_seconds",
    ]
    for (vehicles, strategy), group in sorted(by_key.items()):
        out = {"vehicles": vehicles, "strategy": strategy, "instances": len(group)}
        for metric in metrics:
            values = [float(r[metric]) for r in group if r[metric] not in (None, "")]
            out[f"mean_{metric}"] = round(sum(values) / len(values), 6) if values else None
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a compact multi-day routing experiment grid.")
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--data")
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--vehicles", nargs="*", type=int, default=[3, 5])
    parser.add_argument("--max-events", type=int, default=25)
    parser.add_argument(
        "--selection-rule",
        choices=["time", "spatial", "value"],
        default="time",
        help="Candidate cap rule. 'time' avoids pre-filtering by predicted value.",
    )
    parser.add_argument("--risk-file")
    parser.add_argument("--risk-source", default="bert_chronological_calibrated")
    parser.add_argument("--strategies", nargs="*", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-csv", default="outputs/routing_grid_results.csv")
    parser.add_argument("--summary-csv", default="outputs/routing_grid_summary.csv")
    parser.add_argument("--output-json", default="outputs/routing_grid_results.json")
    args = parser.parse_args()

    cfg = read_json(args.config)
    rows = read_csv_rows(args.data or cfg["data_file"])
    risk_lookup = load_risk_file(args.risk_file, cfg["id_column"])
    dates = select_dates(rows, cfg, args.dates)
    result_rows = []
    skipped = []

    for date_text in dates:
        for vehicle_count in args.vehicles:
            events = build_events(rows, cfg, date_text, risk_lookup, cfg["time_window_hours_default"])
            events = select_candidate_events(events, args.max_events, args.selection_rule)
            if not events:
                skipped.append({"date": date_text, "vehicles": vehicle_count, "reason": "no usable events"})
                continue
            for offset, strategy in enumerate(args.strategies):
                start = time.perf_counter()
                strategy_seed = args.seed + offset if strategy == "random" else args.seed
                routes, depot, shift_start = run_strategy(
                    strategy,
                    events,
                    cfg,
                    vehicle_count,
                    strategy_seed,
                    date_text,
                )
                runtime = time.perf_counter() - start
                metrics = evaluate(routes, events, depot, cfg, shift_start)
                result_rows.append(
                    metric_row(
                        date_text,
                        vehicle_count,
                        strategy,
                        metrics,
                        runtime,
                        len(events),
                        args.risk_source if args.risk_file else "rule_proxy_for_pipeline_testing",
                    )
                )

    summary_rows = summarize(result_rows)
    write_csv(args.output_csv, result_rows)
    write_csv(args.summary_csv, summary_rows)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dates": dates,
                "vehicles": args.vehicles,
                "max_events": args.max_events,
                "selection_rule": args.selection_rule,
                "strategies": args.strategies,
                "skipped": skipped,
                "results": result_rows,
                "summary": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
