import argparse
import csv
import json
import os
import random
import statistics
from collections import defaultdict

from env_inspection_pipeline import (
    build_events,
    haversine_km,
    load_risk_file,
    read_csv_rows,
    read_json,
    selected_set,
)
from run_routing_experiment_grid import select_candidate_events
from run_two_stage_svrptw_experiment import generate_scenarios, run_method


DEFAULT_DATES = [
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
]

METHOD_OFFSETS = {
    "risk_first": 0,
    "deterministic_sir": 1,
    "saa_no_recourse": 2,
    "two_stage_recourse": 3,
    "dro_recourse": 4,
}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_neighborhood(events, radius_km):
    nearest = []
    counts = []
    for i, event in enumerate(events):
        distances = [
            haversine_km((event["lon"], event["lat"]), (other["lon"], other["lat"]))
            for j, other in enumerate(events)
            if i != j
        ]
        nearest.append(min(distances) if distances else 0.0)
        counts.append(sum(distance <= radius_km for distance in distances))
    return nearest, counts


def selected_nearest_distance(events, selected):
    selected_list = sorted(selected)
    output = {}
    for i in selected_list:
        distances = [
            haversine_km(
                (events[i]["lon"], events[i]["lat"]),
                (events[j]["lon"], events[j]["lat"]),
            )
            for j in selected_list
            if i != j
        ]
        output[i] = min(distances) if distances else 0.0
    return output


def mean(values):
    return sum(values) / len(values) if values else 0.0


def residual_cluster_key(event, grid=0.025):
    district = str(event.get("district") or "").strip()
    if district:
        return ("district", district)
    return ("grid", round(event["lon"] / grid), round(event["lat"] / grid))


def build_residual_blocks(rows, cfg, risk_lookup, start_date, end_date):
    dates = sorted(
        {
            str(row.get(cfg["date_column"], "")).strip()
            for row in rows
            if start_date
            <= str(row.get(cfg["date_column"], "")).strip()
            <= end_date
        }
    )
    blocks = []
    for date_text in dates:
        events = build_events(
            rows,
            cfg,
            date_text,
            risk_lookup,
            cfg["time_window_hours_default"],
        )
        labelled = [event for event in events if event.get("label") in (0, 1)]
        if not labelled:
            continue
        residuals = [float(event["label"]) - float(event["risk"]) for event in labelled]
        global_residual = mean(residuals)
        grouped = defaultdict(list)
        for event, residual in zip(labelled, residuals):
            grouped[residual_cluster_key(event)].append(residual - global_residual)
        blocks.append(
            {
                "date": date_text,
                "global_residual": global_residual,
                "cluster_residuals": {
                    key: mean(values) for key, values in grouped.items()
                },
                "observations": len(labelled),
            }
        )
    if not blocks:
        raise ValueError("No labelled calibration blocks were available for scenario generation.")
    return blocks


def generate_residual_block_scenarios(events, scenario_count, seed, blocks):
    rng = random.Random(seed)
    scenarios = []
    for _ in range(scenario_count):
        block = blocks[rng.randrange(len(blocks))]
        scenario = []
        for event in events:
            residual = block["global_residual"] + block["cluster_residuals"].get(
                residual_cluster_key(event), 0.0
            )
            probability = max(0.0, min(1.0, float(event["risk"]) + residual))
            lower = event.get("conformal_lower")
            upper = event.get("conformal_upper")
            if lower is not None:
                probability = max(float(lower), probability)
            if upper is not None:
                probability = min(float(upper), probability)
            scenario.append(1 if rng.random() < probability else 0)
        scenarios.append(scenario)
    return scenarios


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    output = []
    for method, group in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "batches": len(group),
                "mean_positives_found": round(mean([row["positives_found"] for row in group]), 6),
                "mean_positive_coverage": round(mean([row["positive_coverage"] for row in group]), 6),
                "mean_worst_case_objective": round(
                    mean([row["worst_case_objective"] for row in group]), 6
                ),
                "mean_realized_recourse_value": round(
                    mean([row["realized_recourse_value"] for row in group]), 6
                ),
                "mean_positives_per_patrol_hour": round(
                    mean([row["positives_per_patrol_hour"] for row in group]), 6
                ),
                "mean_distance_km": round(mean([row["distance_km"] for row in group]), 6),
                "mean_overtime_hours": round(
                    mean([row["overtime_hours"] for row in group]), 6
                ),
                "mean_expected_unresolved_count": round(
                    mean([row["expected_unresolved_count"] for row in group]), 6
                ),
                "mean_positives_per_100km": round(
                    mean([row["positives_per_100km"] for row in group]), 6
                ),
                "mean_selected_candidate_neighbors_5km": round(
                    mean([row["selected_candidate_neighbors_5km"] for row in group]), 6
                ),
                "mean_selected_nearest_candidate_km": round(
                    mean([row["selected_nearest_candidate_km"] for row in group]), 6
                ),
                "mean_selected_nearest_selected_km": round(
                    mean([row["selected_nearest_selected_km"] for row in group]), 6
                ),
                "mean_clustered_selected_share": round(
                    mean([row["clustered_selected_share"] for row in group]), 6
                ),
                "mean_isolated_high_risk_selected": round(
                    mean([row["isolated_high_risk_selected"] for row in group]), 6
                ),
                "mean_clustered_moderate_risk_selected": round(
                    mean([row["clustered_moderate_risk_selected"] for row in group]), 6
                ),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Event-level audit of routing, realized labels, and spatial complementarity."
    )
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--data")
    parser.add_argument(
        "--risk-file",
        default="outputs/risk_predictions_bert_chrono_public.csv",
    )
    parser.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=40)
    parser.add_argument("--scenarios", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--scenario-mode",
        choices=["independent", "residual_block"],
        default="residual_block",
    )
    parser.add_argument("--residual-start", default="2024-09-01")
    parser.add_argument("--residual-end", default="2024-10-31")
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=list(METHOD_OFFSETS),
        default=["risk_first", "deterministic_sir", "dro_recourse"],
    )
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--remote-capacity-per-team", type=float, default=1.0)
    parser.add_argument("--handoff-capacity-per-team", type=float, default=1.5)
    parser.add_argument("--remote-cost", type=float, default=0.15)
    parser.add_argument("--handoff-cost", type=float, default=0.35)
    parser.add_argument("--unresolved-penalty", type=float, default=1.25)
    parser.add_argument("--dro-radius", type=float, default=0.15)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    cfg = read_json(args.config)
    rows = read_csv_rows(args.data or cfg["data_file"])
    risk_lookup = load_risk_file(args.risk_file, cfg["id_column"])
    residual_blocks = (
        build_residual_blocks(
            rows,
            cfg,
            risk_lookup,
            args.residual_start,
            args.residual_end,
        )
        if args.scenario_mode == "residual_block"
        else []
    )
    params = {
        "remote_capacity_per_team": args.remote_capacity_per_team,
        "handoff_capacity_per_team": args.handoff_capacity_per_team,
        "remote_cost": args.remote_cost,
        "handoff_cost": args.handoff_cost,
        "unresolved_penalty": args.unresolved_penalty,
        "dro_radius": args.dro_radius,
    }

    batch_rows = []
    event_rows = []
    for date_text in args.dates:
        base_events = build_events(
            rows,
            cfg,
            date_text,
            risk_lookup,
            cfg["time_window_hours_default"],
        )
        events = select_candidate_events(base_events, args.max_events, "time")
        if not events:
            continue

        nearest_candidate, candidate_neighbors = candidate_neighborhood(events, args.radius_km)
        risks = sorted(event["risk"] for event in events)
        lower = risks[int(0.25 * (len(risks) - 1))]
        upper = risks[int(0.75 * (len(risks) - 1))]
        scenario_seed = args.seed + args.vehicles * 100 + int(date_text[-2:])
        scenarios = (
            generate_residual_block_scenarios(
                events,
                args.scenarios,
                scenario_seed,
                residual_blocks,
            )
            if args.scenario_mode == "residual_block"
            else generate_scenarios(events, args.scenarios, scenario_seed)
        )

        for method in args.methods:
            method_seed = args.seed + METHOD_OFFSETS[method] * 1009
            routes, metrics, _, _ = run_method(
                method,
                events,
                cfg,
                args.vehicles,
                method_seed,
                date_text,
                scenarios,
                params,
                args.population,
                args.generations,
                "dro_recourse",
            )
            selected = selected_set(routes)
            nearest_selected = selected_nearest_distance(events, selected)
            selected_indices = sorted(selected)
            clustered_selected = [
                i for i in selected_indices if candidate_neighbors[i] >= 1
            ]
            isolated_high = [
                i
                for i in selected_indices
                if candidate_neighbors[i] == 0 and events[i]["risk"] >= upper
            ]
            clustered_moderate = [
                i
                for i in selected_indices
                if candidate_neighbors[i] >= 1 and lower <= events[i]["risk"] < upper
            ]
            distance = float(metrics["distance_km"])
            positives = int(metrics["positives_found"])
            batch_rows.append(
                {
                    "date": date_text,
                    "method": method,
                    "candidate_count": len(events),
                    "selected_count": len(selected_indices),
                    "worst_case_objective": round(
                        float(metrics["worst_case_objective"]), 6
                    ),
                    "realized_recourse_value": round(
                        float(metrics["realized_recourse_value"]), 6
                    ),
                    "positives_found": positives,
                    "positive_coverage": round(float(metrics["positive_coverage"] or 0.0), 6),
                    "positives_per_patrol_hour": round(
                        float(metrics["positives_per_patrol_hour"]), 6
                    ),
                    "distance_km": round(distance, 6),
                    "overtime_hours": round(float(metrics["overtime_hours"]), 6),
                    "expected_unresolved_count": round(
                        float(metrics["expected_unresolved_count"]), 6
                    ),
                    "positives_per_100km": round(100.0 * positives / distance, 6)
                    if distance > 0
                    else 0.0,
                    "selected_candidate_neighbors_5km": round(
                        mean([candidate_neighbors[i] for i in selected_indices]), 6
                    ),
                    "selected_nearest_candidate_km": round(
                        mean([nearest_candidate[i] for i in selected_indices]), 6
                    ),
                    "selected_nearest_selected_km": round(
                        mean([nearest_selected[i] for i in selected_indices]), 6
                    ),
                    "clustered_selected_share": round(
                        len(clustered_selected) / len(selected_indices), 6
                    )
                    if selected_indices
                    else 0.0,
                    "isolated_high_risk_selected": len(isolated_high),
                    "clustered_moderate_risk_selected": len(clustered_moderate),
                }
            )

            for i, event in enumerate(events):
                event_rows.append(
                    {
                        "date": date_text,
                        "method": method,
                        "event_id": event["id"],
                        "risk": round(float(event["risk"]), 8),
                        "label": event["label"],
                        "selected": int(i in selected),
                        "nearest_candidate_km": round(nearest_candidate[i], 6),
                        "candidate_neighbors_5km": candidate_neighbors[i],
                        "nearest_selected_km": round(nearest_selected.get(i, 0.0), 6),
                        "risk_group": "high"
                        if event["risk"] >= upper
                        else ("moderate" if event["risk"] >= lower else "low"),
                        "spatial_group": "clustered"
                        if candidate_neighbors[i] >= 1
                        else "isolated",
                        "category": event["category"],
                        "source": event["source"],
                    }
                )

    summary_rows = summarize(batch_rows)
    write_csv(f"{args.output_prefix}_batches.csv", batch_rows)
    write_csv(f"{args.output_prefix}_events.csv", event_rows)
    write_csv(f"{args.output_prefix}_summary.csv", summary_rows)
    with open(f"{args.output_prefix}_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dates": args.dates,
                "vehicles": args.vehicles,
                "max_events": args.max_events,
                "scenario_count": args.scenarios,
                "scenario_mode": args.scenario_mode,
                "seed": args.seed,
                "methods": args.methods,
                "radius_km": args.radius_km,
                "recourse_parameters": params,
                "residual_start": args.residual_start
                if args.scenario_mode == "residual_block"
                else None,
                "residual_end": args.residual_end
                if args.scenario_mode == "residual_block"
                else None,
                "residual_blocks": len(residual_blocks),
                "global_residual_mean": round(
                    mean([block["global_residual"] for block in residual_blocks]), 8
                )
                if residual_blocks
                else None,
                "global_residual_sd": round(
                    statistics.pstdev(
                        [block["global_residual"] for block in residual_blocks]
                    ),
                    8,
                )
                if residual_blocks
                else None,
                "summary": summary_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    random.seed(2026)
    main()
