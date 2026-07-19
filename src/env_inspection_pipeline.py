import argparse
import csv
import json
import math
import os
import random
import statistics
import time as time_module
from copy import deepcopy
from datetime import datetime, timedelta


DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    if path:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


def read_csv_rows(path):
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                return [repair_row(row) for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, "cannot decode CSV with utf-8-sig, utf-8, or gb18030")


def repair_mojibake(value):
    if not isinstance(value, str):
        return value
    mapping = {
        "ÊÇ": "是",
        "·ñ": "否",
        "³ÈÉ«": "橙色",
        "ºìÉ«": "红色",
        "»ÆÉ«": "黄色",
        "À¶É«": "蓝色",
    }
    if value in mapping:
        return mapping[value]
    return value


def repair_row(row):
    return {repair_mojibake(k): repair_mojibake(v) for k, v in row.items()}


def parse_dt(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def label_to_int(value):
    text = str(value or "").strip()
    if text == "是":
        return 1
    if text == "否":
        return 0
    return None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = int(math.floor((len(ordered) - 1) * q))
    return ordered[pos]


def summarize_numeric(values):
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "p10": round(percentile(clean, 0.10), 4),
        "p25": round(percentile(clean, 0.25), 4),
        "median": round(percentile(clean, 0.50), 4),
        "p75": round(percentile(clean, 0.75), 4),
        "p90": round(percentile(clean, 0.90), 4),
        "min": round(clean[0], 4),
        "max": round(clean[-1], 4),
    }


def count_by(rows, column):
    counts = {}
    for row in rows:
        key = str(row.get(column, "")).strip()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def text_present(row, columns):
    return any(str(row.get(col, "")).strip() for col in columns)


def has_leakage_text(row, cfg):
    keywords = cfg["leakage_keywords"]
    for col in cfg["leakage_sensitive_text_columns"]:
        text = str(row.get(col, ""))
        if any(keyword in text for keyword in keywords):
            return True
    return False


def safe_text(row, cfg):
    parts = []
    for col in cfg["pre_event_text_columns"]:
        text = str(row.get(col, "")).strip()
        if text:
            parts.append(text)
    event_desc = str(row.get("事件描述", "")).strip()
    if event_desc and not has_leakage_text(row, cfg):
        parts.append(event_desc)
    return " ".join(parts)


def rule_priority(row):
    score = 0.0
    joined = " ".join(str(v or "") for v in row.values())
    if "重点" in str(row.get("是否重点时段预警", "")):
        score += 0.20
    if "监测报警" in str(row.get("事件来源大类", "")):
        score += 0.20
    if "报警" in str(row.get("报警类型", "")) or "报警" in str(row.get("任务描述", "")):
        score += 0.15
    if "露天焚烧" in joined:
        score += 0.15
    if "工业企业" in joined:
        score += 0.10
    if "油烟" in joined or "餐饮" in joined:
        score += 0.10
    if "扬尘" in joined or "施工工地" in joined:
        score += 0.10
    if "异常" in joined or "超标" in joined or "污染" in joined:
        score += 0.15
    return min(score, 1.0)


def proxy_risk(row, cfg):
    text = safe_text(row, cfg)
    score = 0.08 + 0.70 * rule_priority(row)
    risk_words = ["异常", "超标", "露天焚烧", "油烟", "扬尘", "黑烟", "污染", "气液比", "未落实", "违规"]
    for word in risk_words:
        if word in text:
            score += 0.035
    return max(0.01, min(score, 0.95))


def load_risk_file(path, id_column):
    if not path:
        return {}
    rows = read_csv_rows(path)
    risk = {}
    for row in rows:
        key = str(row.get(id_column, "")).strip()
        value = parse_float(row.get("risk"))
        if key and value is not None:
            entry = {"risk": max(0.0, min(1.0, value))}
            for col in ("uncertainty_radius", "conformal_radius", "conformal_lower", "conformal_upper"):
                parsed = parse_float(row.get(col))
                if parsed is not None:
                    entry[col] = parsed
            if str(row.get("conformal_bin", "")).strip():
                entry["conformal_bin"] = str(row.get("conformal_bin")).strip()
            risk[key] = entry
    return risk


def haversine_km(a, b):
    lon1, lat1 = a
    lon2, lat2 = b
    radius = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    x = math.sin(dlat / 2) ** 2
    y = math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(x + y))


def travel_hours(a, b, cfg):
    distance = haversine_km(a, b) * cfg["road_detour_factor"]
    return distance / cfg["average_speed_kmph"]


def derive_event_time(row, cfg):
    return parse_dt(row.get(cfg["event_time_column"])) or parse_dt(row.get(cfg["receive_time_column"]))


def derive_deadline(row, event_time, cfg, window_hours):
    explicit = parse_dt(row.get(cfg["deadline_column"]))
    if explicit:
        return explicit
    if event_time:
        return event_time + timedelta(hours=window_hours)
    return None


def profile(args):
    cfg = read_json(args.config)
    rows = read_csv_rows(args.data or cfg["data_file"])
    label_col = cfg["label_column"]
    lon_col = cfg["longitude_column"]
    lat_col = cfg["latitude_column"]
    pre_text_cols = cfg["pre_event_text_columns"]
    all_text_cols = list(dict.fromkeys(pre_text_cols + cfg["leakage_sensitive_text_columns"]))

    valid_coords = []
    deadline_hours = []
    dispatch_like_hours = []
    daily = {}
    leakage_rows = 0

    for row in rows:
        lon = parse_float(row.get(lon_col))
        lat = parse_float(row.get(lat_col))
        if lon is not None and lat is not None and 100 < lon < 110 and 25 < lat < 35:
            valid_coords.append((lon, lat))
        event_time = derive_event_time(row, cfg)
        deadline = parse_dt(row.get(cfg["deadline_column"]))
        if event_time and deadline:
            hours = (deadline - event_time).total_seconds() / 3600
            if 0 <= hours <= 240:
                deadline_hours.append(hours)
        receive = parse_dt(row.get(cfg["receive_time_column"]))
        if event_time and receive:
            hours = (receive - event_time).total_seconds() / 3600
            if 0 <= hours <= 240:
                dispatch_like_hours.append(hours)
        day = str(row.get(cfg["date_column"], "")).strip()
        if day:
            slot = daily.setdefault(day, {"events": 0, "positives": 0})
            slot["events"] += 1
            if label_to_int(row.get(label_col)) == 1:
                slot["positives"] += 1
        if has_leakage_text(row, cfg):
            leakage_rows += 1

    daily_events = [v["events"] for v in daily.values()]
    daily_pos = [v["positives"] for v in daily.values()]
    lons = [c[0] for c in valid_coords]
    lats = [c[1] for c in valid_coords]

    result = {
        "rows": len(rows),
        "columns": len(rows[0]) if rows else 0,
        "valid_coordinate_rows": len(valid_coords),
        "pre_event_text_available_rows": sum(text_present(r, pre_text_cols) for r in rows),
        "any_text_available_rows": sum(text_present(r, all_text_cols) for r in rows),
        "label_distribution": count_by(rows, label_col),
        "secondary_label_distribution": count_by(rows, cfg["secondary_label_column"]),
        "deadline_available_rows": sum(bool(str(r.get(cfg["deadline_column"], "")).strip()) for r in rows),
        "leakage_sensitive_rows": leakage_rows,
        "deadline_hours_from_event": summarize_numeric(deadline_hours),
        "receive_hours_from_event": summarize_numeric(dispatch_like_hours),
        "daily_event_count": summarize_numeric(daily_events),
        "daily_positive_count": summarize_numeric(daily_pos),
        "longitude": summarize_numeric(lons),
        "latitude": summarize_numeric(lats),
        "top_categories": dict(list(count_by(rows, "大类").items())[:12]),
        "top_sources": dict(list(count_by(rows, "事件来源大类").items())[:8]),
    }
    write_json(args.output, result)


def build_events(rows, cfg, date_text, risk_lookup, window_hours):
    events = []
    for row in rows:
        if date_text and str(row.get(cfg["date_column"], "")).strip() != date_text:
            continue
        lon = parse_float(row.get(cfg["longitude_column"]))
        lat = parse_float(row.get(cfg["latitude_column"]))
        if lon is None or lat is None or not (100 < lon < 110 and 25 < lat < 35):
            continue
        event_time = derive_event_time(row, cfg)
        if not event_time:
            continue
        deadline = derive_deadline(row, event_time, cfg, window_hours)
        if not deadline:
            continue
        event_id = str(row.get(cfg["id_column"], "")).strip()
        risk_entry = risk_lookup.get(event_id)
        if isinstance(risk_entry, dict):
            p = risk_entry.get("risk", proxy_risk(row, cfg))
            probability_radius = risk_entry.get("uncertainty_radius")
            conformal_radius = risk_entry.get("conformal_radius")
            conformal_lower = risk_entry.get("conformal_lower")
            conformal_upper = risk_entry.get("conformal_upper")
        else:
            p = risk_entry if risk_entry is not None else proxy_risk(row, cfg)
            probability_radius = None
            conformal_radius = None
            conformal_lower = None
            conformal_upper = None
        s = rule_priority(row)
        label = label_to_int(row.get(cfg["label_column"]))
        value = cfg["alpha_risk"] * p + cfg["beta_rule_priority"] * s
        if probability_radius is None:
            ambiguity = 1.0 - abs(2.0 * max(0.01, min(0.99, p)) - 1.0)
            probability_radius = max(0.005, p * (0.10 + 0.35 * ambiguity))
        reward_radius = min(value, cfg["alpha_risk"] * max(0.0, probability_radius))
        events.append(
            {
                "id": event_id,
                "date": str(row.get(cfg["date_column"], "")).strip(),
                "lon": lon,
                "lat": lat,
                "event_time": event_time,
                "deadline": deadline,
                "risk": p,
                "rule_priority": s,
                "value": value,
                "uncertainty_radius": reward_radius,
                "conformal_radius": conformal_radius,
                "conformal_lower": conformal_lower,
                "conformal_upper": conformal_upper,
                "label": label,
                "category": str(row.get("大类", "")).strip(),
                "source": str(row.get("事件来源大类", "")).strip(),
                "district": str(row.get("区（市）县", "")).strip(),
            }
        )
    return events


def median(values):
    ordered = sorted(values)
    return statistics.median(ordered) if ordered else 0.0


def default_depot(events):
    return (median([e["lon"] for e in events]), median([e["lat"] for e in events]))


def route_schedule(route, events, depot, cfg, shift_start):
    service_h = cfg["service_minutes_default"] / 60
    distance = 0.0
    lateness = 0.0
    current = depot
    time = shift_start
    stops = []
    for idx in route:
        event = events[idx]
        loc = (event["lon"], event["lat"])
        leg_km = haversine_km(current, loc) * cfg["road_detour_factor"]
        distance += leg_km
        time += timedelta(hours=leg_km / cfg["average_speed_kmph"])
        earliest = max(event["event_time"], shift_start)
        if time < earliest:
            time = earliest
        late_h = max(0.0, (time - event["deadline"]).total_seconds() / 3600)
        lateness += late_h
        stops.append(
            {
                "id": event["id"],
                "arrival": time.isoformat(sep=" "),
                "deadline": event["deadline"].isoformat(sep=" "),
                "risk": round(event["risk"], 4),
                "value": round(event["value"], 4),
                "label": event["label"],
                "lateness_hours": round(late_h, 4),
            }
        )
        time += timedelta(hours=service_h)
        current = loc
    if route:
        leg_km = haversine_km(current, depot) * cfg["road_detour_factor"]
        distance += leg_km
        time += timedelta(hours=leg_km / cfg["average_speed_kmph"])
    duration_h = (time - shift_start).total_seconds() / 3600
    overtime = max(0.0, duration_h - cfg["workday_hours"])
    return {
        "stops": stops,
        "distance_km": distance,
        "lateness_hours": lateness,
        "duration_hours": duration_h,
        "overtime_hours": overtime,
    }


def evaluate(routes, events, depot, cfg, shift_start):
    visited = {idx for route in routes for idx in route}
    gross = sum(events[idx]["value"] for idx in visited)
    route_metrics = [route_schedule(route, events, depot, cfg, shift_start) for route in routes]
    distance = sum(r["distance_km"] for r in route_metrics)
    lateness = sum(r["lateness_hours"] for r in route_metrics)
    overtime = sum(r["overtime_hours"] for r in route_metrics)
    missed_high = sum(
        e["risk"] for i, e in enumerate(events) if i not in visited and e["risk"] >= cfg["high_risk_threshold"]
    )
    objective = (
        gross
        - cfg["travel_cost_per_km"] * distance
        - cfg["lateness_penalty_per_hour"] * lateness
        - cfg["overtime_penalty_per_hour"] * overtime
        - cfg["missed_high_risk_penalty"] * missed_high
    )
    positives_found = sum(1 for idx in visited if events[idx]["label"] == 1)
    positives_total = sum(1 for e in events if e["label"] == 1)
    total_work_h = max(0.001, sum(r["duration_hours"] for r in route_metrics))
    return {
        "objective": objective,
        "visited_count": len(visited),
        "unvisited_count": len(events) - len(visited),
        "gross_value": gross,
        "distance_km": distance,
        "lateness_hours": lateness,
        "overtime_hours": overtime,
        "missed_high_risk": missed_high,
        "positives_found": positives_found,
        "positives_total": positives_total,
        "positive_coverage": positives_found / positives_total if positives_total else None,
        "positives_per_patrol_hour": positives_found / total_work_h,
        "vehicle_utilization": total_work_h / (len(routes) * cfg["workday_hours"]) if routes else 0.0,
        "route_metrics": route_metrics,
    }


def decode_order(order, events, depot, cfg, shift_start, vehicles):
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
    return routes


def two_opt_route(route, events, depot, cfg, shift_start):
    if len(route) < 4:
        return route
    best_route = route[:]
    best_score = evaluate([best_route], events, depot, cfg, shift_start)["objective"]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route)):
                candidate = best_route[:i] + list(reversed(best_route[i:j])) + best_route[j:]
                score = evaluate([candidate], events, depot, cfg, shift_start)["objective"]
                if score > best_score:
                    best_route = candidate
                    best_score = score
                    improved = True
    return best_route


def alns_repair(routes, events, depot, cfg, shift_start, rng):
    routes = deepcopy(routes)
    visited = [(vehicle, pos, idx) for vehicle, route in enumerate(routes) for pos, idx in enumerate(route)]
    if not visited:
        return routes
    remove_n = max(1, int(len(visited) * cfg["alns_destroy_rate"]))
    visited_sorted = sorted(visited, key=lambda item: events[item[2]]["value"])
    removal_pool = visited_sorted[: max(remove_n, len(visited_sorted) // 2)]
    to_remove = set(rng.sample(removal_pool, min(remove_n, len(removal_pool))))
    removed = []
    for vehicle in reversed(range(len(routes))):
        new_route = []
        for pos, idx in enumerate(routes[vehicle]):
            if (vehicle, pos, idx) in to_remove:
                removed.append(idx)
            else:
                new_route.append(idx)
        routes[vehicle] = new_route

    visited_ids = {idx for route in routes for idx in route}
    candidates = removed + [
        idx for idx, event in enumerate(events) if idx not in visited_ids and event["risk"] >= cfg["high_risk_threshold"]
    ]
    candidates = sorted(set(candidates), key=lambda idx: events[idx]["value"], reverse=True)
    current_score = evaluate(routes, events, depot, cfg, shift_start)["objective"]
    for idx in candidates:
        best = None
        for vehicle in range(len(routes)):
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
    routes = [two_opt_route(route, events, depot, cfg, shift_start) for route in routes]
    return routes


def sequence_from_routes(routes, events):
    seen = set()
    sequence = []
    for route in routes:
        for idx in route:
            if idx not in seen:
                seen.add(idx)
                sequence.append(idx)
    tail = [idx for idx in range(len(events)) if idx not in seen]
    tail.sort(key=lambda idx: events[idx]["value"], reverse=True)
    return sequence + tail


def order_crossover(parent_a, parent_b, rng):
    n = len(parent_a)
    if n <= 2:
        return parent_a[:]
    left, right = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[left:right] = parent_a[left:right]
    used = set(parent_a[left:right])
    fill = [idx for idx in parent_b if idx not in used]
    fill_pos = [pos for pos, val in enumerate(child) if val is None]
    for pos, val in zip(fill_pos, fill):
        child[pos] = val
    return child


def mutate_order(order, events, rng, rate=0.18):
    child = order[:]
    n = len(child)
    if n < 2:
        return child
    moves = max(1, int(n * rate))
    for _ in range(moves):
        if rng.random() < 0.55:
            i, j = rng.sample(range(n), 2)
            child[i], child[j] = child[j], child[i]
        else:
            i, j = rng.sample(range(n), 2)
            val = child.pop(i)
            child.insert(j, val)
    if rng.random() < 0.35:
        head = child[: max(3, n // 3)]
        head.sort(key=lambda idx: events[idx]["value"] + 0.15 * events[idx]["risk"], reverse=True)
        child[: len(head)] = head
    return child


def selected_set(routes):
    return {idx for route in routes for idx in route}


def solution_distance(sol_a, sol_b):
    set_a = sol_a["selected"]
    set_b = sol_b["selected"]
    union = set_a | set_b
    if not union:
        set_distance = 0.0
    else:
        set_distance = 1.0 - len(set_a & set_b) / len(union)
    pos_a = {idx: pos for pos, idx in enumerate(sol_a["order"])}
    pos_b = {idx: pos for pos, idx in enumerate(sol_b["order"])}
    common = set(pos_a) & set(pos_b)
    if not common:
        order_distance = 1.0
    else:
        denom = max(1, len(sol_a["order"]) - 1)
        order_distance = sum(abs(pos_a[idx] - pos_b[idx]) / denom for idx in common) / len(common)
    return 0.75 * set_distance + 0.25 * order_distance


def hgs_alns(events, cfg, vehicles, seed, shift_date=None, use_alns=True):
    rng = random.Random(seed)
    repair_rng = random.Random(seed + 1000003)
    depot = default_depot(events)
    first_day = shift_date or min(e["event_time"].date() for e in events)
    shift_start = datetime.combine(first_day, datetime.min.time()) + timedelta(hours=cfg["shift_start_hour"])
    n = len(events)
    pop_size = int(cfg.get("hgs_population", 18))
    generations = int(cfg.get("hgs_generations", 35))
    education_repeats = int(cfg.get("hgs_education_repeats", 1))
    elite_size = max(4, pop_size // 4)

    def make_solution(order):
        routes = decode_order(order, events, depot, cfg, shift_start, vehicles)
        score = evaluate(routes, events, depot, cfg, shift_start)["objective"]
        if use_alns:
            for _ in range(max(1, education_repeats)):
                repaired = alns_repair(routes, events, depot, cfg, shift_start, repair_rng)
                repaired_score = evaluate(repaired, events, depot, cfg, shift_start)["objective"]
                if repaired_score > score:
                    routes = repaired
                    score = repaired_score
        order_out = sequence_from_routes(routes, events)
        return {
            "order": order_out,
            "routes": routes,
            "score": score,
            "selected": selected_set(routes),
        }

    base_orders = [
        sorted(range(n), key=lambda i: events[i]["value"], reverse=True),
        sorted(range(n), key=lambda i: (events[i]["deadline"], -events[i]["value"])),
        sorted(range(n), key=lambda i: (events[i]["event_time"], -events[i]["value"])),
        sorted(range(n), key=lambda i: (events[i]["risk"], events[i]["value"]), reverse=True),
    ]
    population = []
    for order in base_orders:
        population.append(make_solution(order))
    while len(population) < pop_size:
        order = list(range(n))
        rng.shuffle(order)
        if rng.random() < 0.55:
            order.sort(key=lambda i: rng.random() - 0.45 * events[i]["risk"] - 0.25 * events[i]["value"])
        population.append(make_solution(order))

    def diversity_scores(pop):
        scores = []
        for i, sol in enumerate(pop):
            others = [solution_distance(sol, pop[j]) for j in range(len(pop)) if j != i]
            scores.append(sum(others) / len(others) if others else 0.0)
        return scores

    def tournament(pop):
        diversities = diversity_scores(pop)
        max_abs = max(1.0, max(abs(sol["score"]) for sol in pop))
        sample = rng.sample(range(len(pop)), min(4, len(pop)))
        return max(sample, key=lambda idx: pop[idx]["score"] / max_abs + 0.12 * diversities[idx])

    best = max(population, key=lambda sol: sol["score"])
    no_improve = 0
    for _ in range(generations):
        p1 = population[tournament(population)]
        p2 = population[tournament(population)]
        child_order = order_crossover(p1["order"], p2["order"], rng)
        child_order = mutate_order(child_order, events, rng)
        child = make_solution(child_order)
        population.append(child)
        if child["score"] > best["score"]:
            best = child
            no_improve = 0
        else:
            no_improve += 1

        if len(population) > pop_size + elite_size:
            diversities = diversity_scores(population)
            ranked = sorted(
                range(len(population)),
                key=lambda idx: (population[idx]["score"], 0.08 * diversities[idx]),
                reverse=True,
            )
            keep = sorted(ranked[:pop_size])
            population = [population[idx] for idx in keep]

        if no_improve > max(15, generations // 2):
            immigrant_count = max(2, pop_size // 5)
            population = sorted(population, key=lambda sol: sol["score"], reverse=True)[: pop_size - immigrant_count]
            for _ in range(immigrant_count):
                order = list(range(n))
                rng.shuffle(order)
                population.append(make_solution(order))
            no_improve = 0

    metrics = evaluate(best["routes"], events, depot, cfg, shift_start)
    return best["routes"], metrics, depot, shift_start


def shift_start_for(events, cfg, date_text):
    shift_date = parse_dt(date_text).date() if parse_dt(date_text) else min(e["event_time"].date() for e in events)
    return datetime.combine(shift_date, datetime.min.time()) + timedelta(hours=cfg["shift_start_hour"])


def route_by_sequence(sequence, events, depot, cfg, shift_start, vehicles):
    return decode_order(sequence, events, depot, cfg, shift_start, vehicles)


def random_strategy(events, cfg, vehicles, seed, date_text):
    rng = random.Random(seed)
    depot = default_depot(events)
    shift_start = shift_start_for(events, cfg, date_text)
    order = list(range(len(events)))
    rng.shuffle(order)
    return route_by_sequence(order, events, depot, cfg, shift_start, vehicles), depot, shift_start


def risk_first_strategy(events, cfg, vehicles, seed, date_text):
    depot = default_depot(events)
    shift_start = shift_start_for(events, cfg, date_text)
    order = sorted(range(len(events)), key=lambda i: events[i]["value"], reverse=True)
    return route_by_sequence(order, events, depot, cfg, shift_start, vehicles), depot, shift_start


def nearest_strategy(events, cfg, vehicles, seed, date_text):
    depot = default_depot(events)
    shift_start = shift_start_for(events, cfg, date_text)
    unvisited = set(range(len(events)))
    routes = [[] for _ in range(vehicles)]
    for vehicle in range(vehicles):
        current = depot
        while unvisited:
            candidate = min(
                unvisited,
                key=lambda i: haversine_km(current, (events[i]["lon"], events[i]["lat"])) - 20 * events[i]["value"],
            )
            trial = deepcopy(routes)
            trial[vehicle].append(candidate)
            if evaluate(trial, events, depot, cfg, shift_start)["objective"] >= evaluate(routes, events, depot, cfg, shift_start)["objective"]:
                routes = trial
                unvisited.remove(candidate)
                current = (events[candidate]["lon"], events[candidate]["lat"])
            else:
                break
    return routes, depot, shift_start


def greedy_insertion_strategy(events, cfg, vehicles, seed, date_text):
    depot = default_depot(events)
    shift_start = shift_start_for(events, cfg, date_text)
    routes = [[] for _ in range(vehicles)]
    remaining = set(range(len(events)))
    current_score = evaluate(routes, events, depot, cfg, shift_start)["objective"]
    while remaining:
        best = None
        for idx in remaining:
            for vehicle in range(vehicles):
                for pos in range(len(routes[vehicle]) + 1):
                    trial = deepcopy(routes)
                    trial[vehicle].insert(pos, idx)
                    score = evaluate(trial, events, depot, cfg, shift_start)["objective"]
                    delta = score - current_score
                    if best is None or delta > best[0]:
                        best = (delta, score, idx, trial)
        if best is None or best[0] <= 0:
            break
        routes = best[3]
        current_score = best[1]
        remaining.remove(best[2])
    return routes, depot, shift_start


def alns_only_strategy(events, cfg, vehicles, seed, date_text):
    rng = random.Random(seed)
    routes, depot, shift_start = risk_first_strategy(events, cfg, vehicles, seed, date_text)
    best_routes = routes
    best_score = evaluate(routes, events, depot, cfg, shift_start)["objective"]
    iterations = max(10, int(cfg.get("alns_iterations", 40)))
    for _ in range(iterations):
        candidate = alns_repair(best_routes, events, depot, cfg, shift_start, rng)
        score = evaluate(candidate, events, depot, cfg, shift_start)["objective"]
        if score > best_score:
            best_routes = candidate
            best_score = score
    return best_routes, depot, shift_start


def run_strategy(name, events, cfg, vehicles, seed, date_text):
    if name == "random":
        return random_strategy(events, cfg, vehicles, seed, date_text)
    if name == "nearest":
        return nearest_strategy(events, cfg, vehicles, seed, date_text)
    if name == "risk_first":
        return risk_first_strategy(events, cfg, vehicles, seed, date_text)
    if name == "greedy_insertion":
        return greedy_insertion_strategy(events, cfg, vehicles, seed, date_text)
    if name == "alns_only":
        return alns_only_strategy(events, cfg, vehicles, seed, date_text)
    if name == "hgs_only":
        shift_date = parse_dt(date_text).date() if parse_dt(date_text) else None
        routes, _, depot, shift_start = hgs_alns(events, cfg, vehicles, seed, shift_date=shift_date, use_alns=False)
        return routes, depot, shift_start
    if name == "hgs_alns":
        shift_date = parse_dt(date_text).date() if parse_dt(date_text) else None
        routes, _, depot, shift_start = hgs_alns(events, cfg, vehicles, seed, shift_date=shift_date, use_alns=True)
        return routes, depot, shift_start
    raise ValueError(f"Unknown strategy: {name}")


def metric_row(strategy, metrics, runtime_seconds):
    row = {
        "strategy": strategy,
        "objective": metrics["objective"],
        "visited_count": metrics["visited_count"],
        "positives_found": metrics["positives_found"],
        "positive_coverage": metrics["positive_coverage"],
        "positives_per_patrol_hour": metrics["positives_per_patrol_hour"],
        "distance_km": metrics["distance_km"],
        "lateness_hours": metrics["lateness_hours"],
        "overtime_hours": metrics["overtime_hours"],
        "vehicle_utilization": metrics["vehicle_utilization"],
        "runtime_seconds": runtime_seconds,
    }
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}


def write_csv(path, rows):
    if not path:
        return
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare(args):
    cfg = read_json(args.config)
    rows = read_csv_rows(args.data or cfg["data_file"])
    risk_lookup = load_risk_file(args.risk_file, cfg["id_column"])
    window_hours = args.window_hours or cfg["time_window_hours_default"]
    events = build_events(rows, cfg, args.date, risk_lookup, window_hours)
    if args.max_events and len(events) > args.max_events:
        events = sorted(events, key=lambda e: e["value"], reverse=True)[: args.max_events]
    if not events:
        raise SystemExit("No usable events found for the requested filters.")
    strategies = args.strategies.split(",")
    result_rows = []
    route_store = {}
    for offset, strategy in enumerate(strategies):
        started = time_module.perf_counter()
        base_seed = args.seed or cfg["random_seed"]
        strategy_seed = base_seed + offset if strategy == "random" else base_seed
        routes, depot, shift_start = run_strategy(strategy, events, cfg, args.vehicles, strategy_seed, args.date)
        runtime = time_module.perf_counter() - started
        metrics = evaluate(routes, events, depot, cfg, shift_start)
        route_store[strategy] = [[events[idx]["id"] for idx in route] for route in routes]
        result_rows.append(metric_row(strategy, metrics, runtime))
    result = {
        "date": args.date,
        "event_count": len(events),
        "vehicles": args.vehicles,
        "window_hours": window_hours,
        "risk_source": "external" if risk_lookup else "rule_proxy_for_pipeline_testing",
        "results": result_rows,
        "routes": route_store,
    }
    write_json(args.output_json, result)
    write_csv(args.output_csv, result_rows)


def solve(args):
    cfg = read_json(args.config)
    rows = read_csv_rows(args.data or cfg["data_file"])
    risk_lookup = load_risk_file(args.risk_file, cfg["id_column"])
    window_hours = args.window_hours or cfg["time_window_hours_default"]
    events = build_events(rows, cfg, args.date, risk_lookup, window_hours)
    if args.max_events and len(events) > args.max_events:
        events = sorted(events, key=lambda e: e["value"], reverse=True)[: args.max_events]
    if not events:
        raise SystemExit("No usable events found for the requested filters.")
    vehicles = args.vehicles or cfg["vehicle_counts"][1]
    shift_date = parse_dt(args.date).date() if parse_dt(args.date) else None
    routes, metrics, depot, shift_start = hgs_alns(
        events, cfg, vehicles, args.seed or cfg["random_seed"], shift_date=shift_date, use_alns=True
    )
    output_routes = []
    for idx, route_metric in enumerate(metrics.pop("route_metrics"), start=1):
        output_routes.append(
            {
                "vehicle": idx,
                "stops": route_metric["stops"],
                "distance_km": round(route_metric["distance_km"], 4),
                "duration_hours": round(route_metric["duration_hours"], 4),
                "lateness_hours": round(route_metric["lateness_hours"], 4),
                "overtime_hours": round(route_metric["overtime_hours"], 4),
            }
        )
    result = {
        "date": args.date,
        "event_count": len(events),
        "vehicles": vehicles,
        "window_hours": window_hours,
        "service_minutes": cfg["service_minutes_default"],
        "depot": {"lon": depot[0], "lat": depot[1]},
        "shift_start": shift_start.isoformat(sep=" "),
        "risk_source": "external" if risk_lookup else "rule_proxy_for_pipeline_testing",
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "routes": output_routes,
    }
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description="Environmental inspection risk and selective VRPTW pipeline")
    sub = parser.add_subparsers(required=True)

    p_profile = sub.add_parser("profile", help="profile the dispatch CSV")
    p_profile.add_argument("--config", default="config/defaults.json")
    p_profile.add_argument("--data")
    p_profile.add_argument("--output")
    p_profile.set_defaults(func=profile)

    p_solve = sub.add_parser("solve", help="solve a daily selective VRPTW prototype")
    p_solve.add_argument("--config", default="config/defaults.json")
    p_solve.add_argument("--data")
    p_solve.add_argument("--date", required=True)
    p_solve.add_argument("--risk-file")
    p_solve.add_argument("--vehicles", type=int)
    p_solve.add_argument("--window-hours", type=int)
    p_solve.add_argument("--max-events", type=int, default=80)
    p_solve.add_argument("--seed", type=int)
    p_solve.add_argument("--output")
    p_solve.set_defaults(func=solve)

    p_compare = sub.add_parser("compare", help="compare routing benchmark algorithms")
    p_compare.add_argument("--config", default="config/defaults.json")
    p_compare.add_argument("--data")
    p_compare.add_argument("--date", required=True)
    p_compare.add_argument("--risk-file")
    p_compare.add_argument("--vehicles", type=int, default=3)
    p_compare.add_argument("--window-hours", type=int)
    p_compare.add_argument("--max-events", type=int, default=40)
    p_compare.add_argument("--seed", type=int)
    p_compare.add_argument(
        "--strategies",
        default="random,nearest,risk_first,greedy_insertion,alns_only,hgs_only,hgs_alns",
    )
    p_compare.add_argument("--output-json", default="outputs/algorithm_comparison.json")
    p_compare.add_argument("--output-csv", default="outputs/algorithm_comparison.csv")
    p_compare.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
