import argparse
import csv
import json
import math
import os
import random
import time
from copy import deepcopy

from env_inspection_pipeline import (
    alns_repair,
    build_events,
    default_depot,
    evaluate,
    hgs_alns,
    load_risk_file,
    mutate_order,
    order_crossover,
    parse_dt,
    read_csv_rows,
    read_json,
    route_schedule,
    selected_set,
    sequence_from_routes,
    shift_start_for,
    solution_distance,
)
from run_routing_experiment_grid import select_candidate_events


DEFAULT_DATES = [
    "2024-12-01",
    "2024-12-02",
    "2024-12-17",
    "2024-12-23",
    "2024-12-26",
    "2024-12-31",
]


METHODS = [
    "risk_first",
    "deterministic_sir",
    "saa_no_recourse",
    "two_stage_recourse",
    "dro_recourse",
]


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_scenarios(events, scenario_count, seed):
    rng = random.Random(seed)
    scenarios = []
    for _ in range(scenario_count):
        scenarios.append([1 if rng.random() < max(0.0, min(1.0, e["risk"])) else 0 for e in events])
    return scenarios


def exposure(event):
    return 1.0 + event.get("rule_priority", 0.0)


def fractional_follow_up(unvisited_true, events, capacity):
    remaining = max(0.0, float(capacity))
    assigned = []
    for idx in sorted(unvisited_true, key=lambda item: exposure(events[item]), reverse=True):
        if remaining <= 1e-12:
            break
        frac = min(1.0, remaining)
        assigned.append((idx, frac))
        remaining -= frac
    return assigned


def worst_case_expectation(values, radius):
    """Worst-case mean over a total-variation ball around the empirical distribution."""
    if not values:
        return 0.0
    eps = max(0.0, min(float(radius), 1.0))
    n = len(values)
    weights = [1.0 / n] * n
    order_low = sorted(range(n), key=lambda i: values[i])
    order_high = sorted(range(n), key=lambda i: values[i], reverse=True)
    low_pos = 0
    high_pos = 0
    remaining = eps
    while remaining > 1e-12 and low_pos < n and high_pos < n:
        low = order_low[low_pos]
        high = order_high[high_pos]
        if values[low] >= values[high] - 1e-12:
            break
        add_cap = 1.0 - weights[low]
        take_cap = weights[high]
        transfer = min(remaining, add_cap, take_cap)
        if transfer <= 1e-12:
            if add_cap <= 1e-12:
                low_pos += 1
            if take_cap <= 1e-12:
                high_pos += 1
            continue
        weights[low] += transfer
        weights[high] -= transfer
        remaining -= transfer
        if weights[low] >= 1.0 - 1e-12:
            low_pos += 1
        if weights[high] <= 1e-12:
            high_pos += 1
    return sum(w * v for w, v in zip(weights, values))


def route_costs(routes, events, depot, cfg, shift_start):
    metrics = [route_schedule(route, events, depot, cfg, shift_start) for route in routes]
    distance = sum(r["distance_km"] for r in metrics)
    lateness = sum(r["lateness_hours"] for r in metrics)
    overtime = sum(r["overtime_hours"] for r in metrics)
    cost = (
        cfg["travel_cost_per_km"] * distance
        + cfg["lateness_penalty_per_hour"] * lateness
        + cfg["overtime_penalty_per_hour"] * overtime
    )
    duration = sum(r["duration_hours"] for r in metrics)
    return {
        "distance_km": distance,
        "lateness_hours": lateness,
        "overtime_hours": overtime,
        "route_cost": cost,
        "duration_hours": duration,
        "route_metrics": metrics,
    }


def recourse_score_for_scenario(visited, events, scenario, vehicles, params, model):
    selected_reward = 0.0
    unvisited_true = []
    for idx, event in enumerate(events):
        if not scenario[idx]:
            continue
        if idx in visited:
            selected_reward += exposure(event)
        else:
            unvisited_true.append(idx)

    if model == "deterministic":
        return 0.0, {
            "selected_reward": selected_reward,
            "remote_count": 0,
            "handoff_count": 0,
            "unresolved_count": 0,
            "recourse_cost": 0.0,
        }

    if model == "saa_no_recourse":
        unresolved_weight = sum(exposure(events[idx]) for idx in unvisited_true)
        recourse_cost = params["unresolved_penalty"] * unresolved_weight
        return selected_reward - recourse_cost, {
            "selected_reward": selected_reward,
            "remote_count": 0,
            "handoff_count": 0,
            "unresolved_count": len(unvisited_true),
            "recourse_cost": recourse_cost,
        }

    remote_cap = max(0.0, params["remote_capacity_per_team"] * vehicles)
    handoff_cap = max(0.0, params["handoff_capacity_per_team"] * vehicles)
    remote = fractional_follow_up(unvisited_true, events, remote_cap)
    remote_fraction = {idx: frac for idx, frac in remote}
    remaining_after_remote = [
        idx for idx in unvisited_true if remote_fraction.get(idx, 0.0) < 1.0 - 1e-12
    ]
    handoff_remaining = max(0.0, handoff_cap)
    handoff = []
    for idx in sorted(remaining_after_remote, key=lambda item: exposure(events[item]), reverse=True):
        if handoff_remaining <= 1e-12:
            break
        max_frac = 1.0 - remote_fraction.get(idx, 0.0)
        frac = min(max_frac, handoff_remaining)
        if frac > 1e-12:
            handoff.append((idx, frac))
            handoff_remaining -= frac
    handoff_fraction = {idx: frac for idx, frac in handoff}
    unresolved = []
    for idx in unvisited_true:
        remaining_frac = 1.0 - remote_fraction.get(idx, 0.0) - handoff_fraction.get(idx, 0.0)
        if remaining_frac > 1e-12:
            unresolved.append((idx, remaining_frac))
    remote_cost = params["remote_cost"] * sum(exposure(events[idx]) * frac for idx, frac in remote)
    handoff_cost = params["handoff_cost"] * sum(exposure(events[idx]) * frac for idx, frac in handoff)
    unresolved_cost = params["unresolved_penalty"] * sum(exposure(events[idx]) * frac for idx, frac in unresolved)
    recourse_cost = remote_cost + handoff_cost + unresolved_cost
    return selected_reward - recourse_cost, {
        "selected_reward": selected_reward,
        "remote_count": sum(frac for _, frac in remote),
        "handoff_count": sum(frac for _, frac in handoff),
        "unresolved_count": sum(frac for _, frac in unresolved),
        "recourse_cost": recourse_cost,
    }


def score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, model):
    visited = selected_set(routes)
    costs = route_costs(routes, events, depot, cfg, shift_start)
    if model == "deterministic":
        metrics = evaluate(routes, events, depot, cfg, shift_start)
        out = {
            "objective": metrics["objective"],
            "expected_selected_reward": metrics["gross_value"],
            "expected_recourse_cost": 0.0,
            "expected_remote_count": 0.0,
            "expected_handoff_count": 0.0,
            "expected_unresolved_count": metrics["missed_high_risk"],
        }
    else:
        scenario_values = []
        selected_rewards = []
        remote_counts = []
        handoff_counts = []
        unresolved_counts = []
        recourse_costs = []
        for scenario in scenarios:
            value, details = recourse_score_for_scenario(visited, events, scenario, vehicles, params, model)
            scenario_values.append(value)
            selected_rewards.append(details["selected_reward"])
            remote_counts.append(details["remote_count"])
            handoff_counts.append(details["handoff_count"])
            unresolved_counts.append(details["unresolved_count"])
            recourse_costs.append(details["recourse_cost"])
        nominal_value = sum(scenario_values) / len(scenario_values)
        robust_value = worst_case_expectation(scenario_values, params.get("dro_radius", 0.0))
        objective_value = robust_value if model == "dro_recourse" else nominal_value
        out = {
            "objective": objective_value - costs["route_cost"],
            "nominal_objective": nominal_value - costs["route_cost"],
            "worst_case_objective": robust_value - costs["route_cost"],
            "expected_selected_reward": sum(selected_rewards) / len(selected_rewards),
            "expected_recourse_cost": sum(recourse_costs) / len(recourse_costs),
            "expected_remote_count": sum(remote_counts) / len(remote_counts),
            "expected_handoff_count": sum(handoff_counts) / len(handoff_counts),
            "expected_unresolved_count": sum(unresolved_counts) / len(unresolved_counts),
        }

    labels = [1 if e["label"] == 1 else 0 for e in events]
    realized_value, realized_details = recourse_score_for_scenario(
        visited, events, labels, vehicles, params, "two_stage_recourse"
    )
    positives_found = sum(labels[idx] for idx in visited)
    positives_total = sum(labels)
    total_work = max(0.001, costs["duration_hours"])
    out.update(
        {
            "visited_count": len(visited),
            "positives_found": positives_found,
            "positives_total": positives_total,
            "positive_coverage": positives_found / positives_total if positives_total else None,
            "positives_per_patrol_hour": positives_found / total_work,
            "distance_km": costs["distance_km"],
            "lateness_hours": costs["lateness_hours"],
            "overtime_hours": costs["overtime_hours"],
            "vehicle_utilization": total_work / (vehicles * cfg["workday_hours"]) if vehicles else 0.0,
            "realized_recourse_value": realized_value - costs["route_cost"],
            "realized_remote_count": realized_details["remote_count"],
            "realized_handoff_count": realized_details["handoff_count"],
            "realized_unresolved_count": realized_details["unresolved_count"],
            "realized_recourse_cost": realized_details["recourse_cost"],
        }
    )
    out.setdefault("nominal_objective", out["objective"])
    out.setdefault("worst_case_objective", out["objective"])
    return out


def decode_order(order, events, depot, cfg, shift_start, vehicles, scenarios, params, model):
    routes = [[] for _ in range(vehicles)]
    current_score = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, model)["objective"]
    for idx in order:
        best = None
        for vehicle in range(vehicles):
            for pos in range(len(routes[vehicle]) + 1):
                trial = deepcopy(routes)
                trial[vehicle].insert(pos, idx)
                score = score_routes(
                    trial, events, depot, cfg, shift_start, vehicles, scenarios, params, model
                )["objective"]
                delta = score - current_score
                if best is None or delta > best[0]:
                    best = (delta, score, trial)
        if best and best[0] > 1e-9:
            routes = best[2]
            current_score = best[1]
    return routes


def recourse_repair(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, model, rng):
    routes = deepcopy(routes)
    visited = [(vehicle, pos, idx) for vehicle, route in enumerate(routes) for pos, idx in enumerate(route)]
    if visited:
        remove_n = max(1, int(len(visited) * cfg.get("alns_destroy_rate", 0.2)))
        low_marginal = sorted(visited, key=lambda item: events[item[2]]["risk"])[: max(remove_n, len(visited) // 2)]
        to_remove = set(rng.sample(low_marginal, min(remove_n, len(low_marginal))))
        for vehicle in reversed(range(len(routes))):
            kept = []
            for pos, idx in enumerate(routes[vehicle]):
                if (vehicle, pos, idx) not in to_remove:
                    kept.append(idx)
            routes[vehicle] = kept

    visited_now = selected_set(routes)
    candidates = [
        idx
        for idx, event in enumerate(events)
        if idx not in visited_now and event["risk"] >= max(0.12, cfg["high_risk_threshold"] * 0.5)
    ]
    candidates = sorted(candidates, key=lambda idx: (events[idx]["risk"], events[idx]["value"]), reverse=True)[:20]
    current_score = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, model)["objective"]
    for idx in candidates:
        best = None
        for vehicle in range(vehicles):
            for pos in range(len(routes[vehicle]) + 1):
                trial = deepcopy(routes)
                trial[vehicle].insert(pos, idx)
                score = score_routes(
                    trial, events, depot, cfg, shift_start, vehicles, scenarios, params, model
                )["objective"]
                delta = score - current_score
                if best is None or delta > best[0]:
                    best = (delta, score, trial)
        if best and best[0] > 1e-9:
            routes = best[2]
            current_score = best[1]
    return routes


def population_search(events, cfg, vehicles, seed, date_text, scenarios, params, model, population, generations):
    rng = random.Random(seed)
    repair_rng = random.Random(seed + 9901)
    depot = default_depot(events)
    shift_start = shift_start_for(events, cfg, date_text)
    n = len(events)
    elite_size = max(3, population // 4)

    def make_solution(order):
        routes = decode_order(order, events, depot, cfg, shift_start, vehicles, scenarios, params, model)
        routes = recourse_repair(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, model, repair_rng)
        score = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, model)["objective"]
        return {
            "order": sequence_from_routes(routes, events),
            "routes": routes,
            "score": score,
            "selected": selected_set(routes),
        }

    def make_route_solution(routes):
        score = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, model)["objective"]
        return {
            "order": sequence_from_routes(routes, events),
            "routes": deepcopy(routes),
            "score": score,
            "selected": selected_set(routes),
        }

    shift_date = parse_dt(date_text).date() if parse_dt(date_text) else None
    warm_routes, _, _, _ = hgs_alns(events, cfg, vehicles, seed, shift_date=shift_date, use_alns=True)
    base_orders = [
        sequence_from_routes(warm_routes, events),
        sorted(range(n), key=lambda i: events[i]["risk"], reverse=True),
        sorted(range(n), key=lambda i: events[i]["value"], reverse=True),
        sorted(range(n), key=lambda i: (events[i]["deadline"], -events[i]["risk"])),
        sorted(range(n), key=lambda i: (events[i]["event_time"], -events[i]["risk"])),
    ]
    pop = [make_route_solution(warm_routes)]
    pop.extend(make_solution(order) for order in base_orders)
    while len(pop) < population:
        order = list(range(n))
        rng.shuffle(order)
        if rng.random() < 0.6:
            order.sort(key=lambda i: rng.random() - 0.55 * events[i]["risk"] - 0.15 * events[i]["rule_priority"])
        pop.append(make_solution(order))

    def diversity_scores(solutions):
        out = []
        for i, sol in enumerate(solutions):
            others = [solution_distance(sol, solutions[j]) for j in range(len(solutions)) if j != i]
            out.append(sum(others) / len(others) if others else 0.0)
        return out

    def tournament(solutions):
        diversities = diversity_scores(solutions)
        scale = max(1.0, max(abs(sol["score"]) for sol in solutions))
        sample = rng.sample(range(len(solutions)), min(4, len(solutions)))
        return max(sample, key=lambda idx: solutions[idx]["score"] / scale + 0.10 * diversities[idx])

    best = max(pop, key=lambda sol: sol["score"])
    for _ in range(generations):
        p1 = pop[tournament(pop)]
        p2 = pop[tournament(pop)]
        child_order = order_crossover(p1["order"], p2["order"], rng)
        child_order = mutate_order(child_order, events, rng, rate=0.12)
        child = make_solution(child_order)
        pop.append(child)
        if child["score"] > best["score"]:
            best = child
        if len(pop) > population + elite_size:
            div = diversity_scores(pop)
            keep = sorted(
                range(len(pop)),
                key=lambda idx: (pop[idx]["score"], 0.07 * div[idx]),
                reverse=True,
            )[:population]
            pop = [pop[idx] for idx in sorted(keep)]
    metrics = score_routes(best["routes"], events, depot, cfg, shift_start, vehicles, scenarios, params, model)
    return best["routes"], metrics, depot, shift_start


def risk_first(routes_events, cfg, vehicles, date_text):
    events = routes_events
    depot = default_depot(events)
    shift_start = shift_start_for(events, cfg, date_text)
    order = sorted(range(len(events)), key=lambda i: events[i]["risk"], reverse=True)
    routes = [[] for _ in range(vehicles)]
    current_score = evaluate(routes, events, depot, cfg, shift_start)["objective"]
    for idx in order:
        best = None
        for vehicle in range(vehicles):
            for pos in range(len(routes[vehicle]) + 1):
                trial = deepcopy(routes)
                trial[vehicle].insert(pos, idx)
                score = evaluate(trial, events, depot, cfg, shift_start)["objective"]
                delta = score - current_score
                if best is None or delta > best[0]:
                    best = (delta, score, trial)
        if best and best[0] > 0:
            routes = best[2]
            current_score = best[1]
    return routes, depot, shift_start


def run_method(method, events, cfg, vehicles, seed, date_text, scenarios, params, population, generations, evaluation_model):
    if method == "risk_first":
        routes, depot, shift_start = risk_first(events, cfg, vehicles, date_text)
        metrics = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, evaluation_model)
        return routes, metrics, depot, shift_start
    if method == "deterministic_sir":
        shift_date = parse_dt(date_text).date() if parse_dt(date_text) else None
        routes, _, depot, shift_start = hgs_alns(events, cfg, vehicles, seed, shift_date=shift_date, use_alns=True)
        metrics = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, evaluation_model)
        return routes, metrics, depot, shift_start
    if method == "saa_no_recourse":
        routes, _, depot, shift_start = population_search(
            events, cfg, vehicles, seed, date_text, scenarios, params, "saa_no_recourse", population, generations
        )
        metrics = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, evaluation_model)
        return routes, metrics, depot, shift_start
    if method == "two_stage_recourse":
        routes, _, depot, shift_start = population_search(
            events, cfg, vehicles, seed, date_text, scenarios, params, "two_stage_recourse", population, generations
        )
        metrics = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, evaluation_model)
        return routes, metrics, depot, shift_start
    if method == "dro_recourse":
        routes, _, depot, shift_start = population_search(
            events, cfg, vehicles, seed, date_text, scenarios, params, "dro_recourse", population, generations
        )
        metrics = score_routes(routes, events, depot, cfg, shift_start, vehicles, scenarios, params, evaluation_model)
        return routes, metrics, depot, shift_start
    raise ValueError(f"Unknown method: {method}")


def round_row(row):
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in row.items()}


def summarize(rows):
    by_key = {}
    for row in rows:
        by_key.setdefault((row["vehicles"], row["method"]), []).append(row)
    metrics = [
        "objective",
        "nominal_objective",
        "worst_case_objective",
        "realized_recourse_value",
        "positives_found",
        "positive_coverage",
        "positives_per_patrol_hour",
        "distance_km",
        "overtime_hours",
        "expected_remote_count",
        "expected_handoff_count",
        "expected_unresolved_count",
        "realized_unresolved_count",
        "runtime_seconds",
    ]
    summary = []
    for (vehicles, method), group in sorted(by_key.items()):
        out = {"vehicles": vehicles, "method": method, "instances": len(group)}
        for metric in metrics:
            vals = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            out[f"mean_{metric}"] = round(sum(vals) / len(vals), 6) if vals else None
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Two-stage SAA selective VRPTW experiment with recourse.")
    parser.add_argument("--config", default="config/defaults.json")
    parser.add_argument("--data")
    parser.add_argument(
        "--risk-file",
        default="outputs/risk_predictions_bert_chrono_public.csv",
    )
    parser.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    parser.add_argument("--vehicles", nargs="*", type=int, default=[3, 5])
    parser.add_argument("--max-events", type=int, default=35)
    parser.add_argument("--selection-rule", choices=["time", "spatial", "value"], default="time")
    parser.add_argument("--scenarios", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--remote-capacity-per-team", type=float, default=1.0)
    parser.add_argument("--handoff-capacity-per-team", type=float, default=1.5)
    parser.add_argument("--remote-cost", type=float, default=0.15)
    parser.add_argument("--handoff-cost", type=float, default=0.35)
    parser.add_argument("--unresolved-penalty", type=float, default=1.25)
    parser.add_argument("--dro-radius", type=float, default=0.15)
    parser.add_argument(
        "--evaluation-model",
        choices=["two_stage_recourse", "dro_recourse"],
        default="two_stage_recourse",
    )
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=METHODS)
    parser.add_argument("--output-csv", default="outputs/two_stage_svrptw_results.csv")
    parser.add_argument("--summary-csv", default="outputs/two_stage_svrptw_summary.csv")
    parser.add_argument("--output-json", default="outputs/two_stage_svrptw_results.json")
    args = parser.parse_args()

    cfg = read_json(args.config)
    rows = read_csv_rows(args.data or cfg["data_file"])
    risk_lookup = load_risk_file(args.risk_file, cfg["id_column"])
    params = {
        "remote_capacity_per_team": args.remote_capacity_per_team,
        "handoff_capacity_per_team": args.handoff_capacity_per_team,
        "remote_cost": args.remote_cost,
        "handoff_cost": args.handoff_cost,
        "unresolved_penalty": args.unresolved_penalty,
        "dro_radius": args.dro_radius,
    }

    result_rows = []
    skipped = []
    for date_text in args.dates:
        base_events = build_events(rows, cfg, date_text, risk_lookup, cfg["time_window_hours_default"])
        events = select_candidate_events(base_events, args.max_events, args.selection_rule)
        if not events:
            skipped.append({"date": date_text, "reason": "no usable events"})
            continue
        for vehicles in args.vehicles:
            scenarios = generate_scenarios(events, args.scenarios, args.seed + vehicles * 100 + int(date_text[-2:]))
            for offset, method in enumerate(args.methods):
                start = time.perf_counter()
                routes, metrics, _, _ = run_method(
                    method,
                    events,
                    cfg,
                    vehicles,
                    args.seed + offset * 1009,
                    date_text,
                    scenarios,
                    params,
                    args.population,
                    args.generations,
                    args.evaluation_model,
                )
                runtime = time.perf_counter() - start
                row = {
                    "date": date_text,
                    "vehicles": vehicles,
                    "method": method,
                    "event_count": len(events),
                    "scenario_count": args.scenarios,
                    "visited_count": metrics["visited_count"],
                        "objective": metrics["objective"],
                        "nominal_objective": metrics["nominal_objective"],
                        "worst_case_objective": metrics["worst_case_objective"],
                        "realized_recourse_value": metrics["realized_recourse_value"],
                    "expected_selected_reward": metrics["expected_selected_reward"],
                    "expected_recourse_cost": metrics["expected_recourse_cost"],
                    "expected_remote_count": metrics["expected_remote_count"],
                    "expected_handoff_count": metrics["expected_handoff_count"],
                    "expected_unresolved_count": metrics["expected_unresolved_count"],
                    "positives_found": metrics["positives_found"],
                    "positives_total": metrics["positives_total"],
                    "positive_coverage": metrics["positive_coverage"],
                    "positives_per_patrol_hour": metrics["positives_per_patrol_hour"],
                    "distance_km": metrics["distance_km"],
                    "lateness_hours": metrics["lateness_hours"],
                    "overtime_hours": metrics["overtime_hours"],
                    "vehicle_utilization": metrics["vehicle_utilization"],
                    "realized_remote_count": metrics["realized_remote_count"],
                    "realized_handoff_count": metrics["realized_handoff_count"],
                    "realized_unresolved_count": metrics["realized_unresolved_count"],
                    "realized_recourse_cost": metrics["realized_recourse_cost"],
                    "runtime_seconds": runtime,
                }
                result_rows.append(round_row(row))

    summary_rows = summarize(result_rows)
    write_csv(args.output_csv, result_rows)
    write_csv(args.summary_csv, summary_rows)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dates": args.dates,
                "vehicles": args.vehicles,
                "max_events": args.max_events,
                "selection_rule": args.selection_rule,
            "scenario_count": args.scenarios,
            "dro_radius": args.dro_radius,
            "evaluation_model": args.evaluation_model,
            "params": params,
                "methods": args.methods,
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
