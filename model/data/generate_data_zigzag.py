import numpy as np
import pandas as pd
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

ACTION_MAP = {
    0: "hover",
    1: "forward",
    2: "backward",
    3: "left",
    4: "right",
    5: "yaw_left",
    6: "yaw_right",
    7: "up",
    8: "down",
}

NUM_SAMPLES = 50000

MIN_DIST = 100.0
MAX_DIST = 2000.0

OBSTACLE_CRITICAL = 120.0
OBSTACLE_NEAR = 200.0
OBSTACLE_CLOSE = 400.0
SAFE_DISTANCE = 600.0

VERTICAL_CRITICAL_DOWN = 50.0
VERTICAL_NEAR_DOWN = 150.0
VERTICAL_SAFE_DOWN = 300.0

VERTICAL_CRITICAL_UP = 100.0
VERTICAL_NEAR_UP = 200.0
VERTICAL_SAFE_UP = 400.0


def front_state(d):
    if d < OBSTACLE_CRITICAL:
        return "critical"
    elif d < OBSTACLE_NEAR:
        return "near"
    elif d < OBSTACLE_CLOSE:
        return "close"
    elif d < SAFE_DISTANCE:
        return "approaching"
    else:
        return "clear"


def side_state(d):
    if d < OBSTACLE_CRITICAL:
        return "critical"
    elif d < OBSTACLE_NEAR:
        return "near"
    else:
        return "safe"


def vertical_down_state(d):
    if d < VERTICAL_CRITICAL_DOWN:
        return "critical"
    elif d < VERTICAL_NEAR_DOWN:
        return "near"
    elif d < VERTICAL_SAFE_DOWN:
        return "approaching"
    else:
        return "safe"


def vertical_up_state(d):
    if d < VERTICAL_CRITICAL_UP:
        return "critical"
    elif d < VERTICAL_NEAR_UP:
        return "near"
    elif d < VERTICAL_SAFE_UP:
        return "approaching"
    else:
        return "safe"


def apply_expert_rules(
    dist_front, dist_back, dist_left, dist_right, dist_down, dist_up
):
    down_state = vertical_down_state(dist_down)
    up_state = vertical_up_state(dist_up)

    if down_state == "critical":
        return 7
    if up_state == "critical":
        return 8

    if down_state == "near" and up_state in ["safe", "approaching"]:
        return 7
    if up_state == "near" and down_state in ["safe", "approaching"]:
        return 8

    f_state = front_state(dist_front)
    l_state = side_state(dist_left)
    r_state = side_state(dist_right)

    if l_state == "critical" and dist_right > OBSTACLE_CRITICAL + 50:
        return 4
    if r_state == "critical" and dist_left > OBSTACLE_CRITICAL + 50:
        return 3

    dead_end = (
        f_state in ("critical", "near")
        and l_state == "critical"
        and r_state == "critical"
    )
    if dead_end:
        return 5

    if f_state == "critical":
        if l_state != "critical" and r_state == "critical":
            return 5
        elif r_state != "critical" and l_state == "critical":
            return 6
        elif l_state != "critical" and r_state != "critical":
            return 5 if dist_left > dist_right else 6
        else:
            if dist_back > OBSTACLE_NEAR:
                return 2
            else:
                return 5

    if f_state == "near":
        if l_state == "near" and r_state == "near":
            return 1
        if l_state != "critical" and r_state == "critical":
            return 5
        elif r_state != "critical" and l_state == "critical":
            return 6
        else:
            return 5 if dist_left > dist_right else 6

    if f_state == "close":
        side_diff = abs(dist_left - dist_right)
        if side_diff > 200.0:
            return 5 if dist_left > dist_right else 6
        else:
            return 1

    if f_state == "approaching":
        side_diff = abs(dist_left - dist_right)
        if side_diff > 300.0:
            return 5 if dist_left > dist_right else 6
        else:
            return 1

    if f_state == "clear":
        if l_state == "critical":
            return 4
        if r_state == "critical":
            return 3
        return 1

    return 1


def sample_in_range(lo, hi):
    return random.uniform(lo, hi)


def generate_scenario(scenario_type):
    if scenario_type == "ground_critical":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_down": sample_in_range(10, VERTICAL_CRITICAL_DOWN),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "ceiling_critical":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, MAX_DIST),
            "dist_up": sample_in_range(30, VERTICAL_CRITICAL_UP),
        }

    elif scenario_type == "ground_near":
        return {
            "dist_front": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_down": sample_in_range(
                VERTICAL_CRITICAL_DOWN + 5, VERTICAL_NEAR_DOWN
            ),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "ceiling_near":
        return {
            "dist_front": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, MAX_DIST),
            "dist_up": sample_in_range(VERTICAL_CRITICAL_UP + 5, VERTICAL_NEAR_UP),
        }

    elif scenario_type == "vertical_squeeze":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_down": sample_in_range(VERTICAL_NEAR_DOWN, VERTICAL_SAFE_DOWN),
            "dist_up": sample_in_range(VERTICAL_CRITICAL_UP + 10, VERTICAL_NEAR_UP),
        }

    elif scenario_type == "ground_with_front_obstacle":
        return {
            "dist_front": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_NEAR, SAFE_DISTANCE),
            "dist_right": sample_in_range(OBSTACLE_NEAR, SAFE_DISTANCE),
            "dist_down": sample_in_range(VERTICAL_CRITICAL_DOWN, VERTICAL_NEAR_DOWN),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "ceiling_with_front_obstacle":
        return {
            "dist_front": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_NEAR, SAFE_DISTANCE),
            "dist_right": sample_in_range(OBSTACLE_NEAR, SAFE_DISTANCE),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, MAX_DIST),
            "dist_up": sample_in_range(VERTICAL_CRITICAL_UP, VERTICAL_NEAR_UP),
        }

    elif scenario_type == "open_space":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_right": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_down": sample_in_range(400, 600),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "tight_corridor":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(150, OBSTACLE_NEAR + 20),
            "dist_right": sample_in_range(150, OBSTACLE_NEAR + 20),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "very_tight_corridor":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(130, 180),
            "dist_right": sample_in_range(130, 180),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "left_wall_critical":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_right": sample_in_range(OBSTACLE_CRITICAL + 100, MAX_DIST),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "right_wall_critical":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CRITICAL + 100, MAX_DIST),
            "dist_right": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "left_wall_near_comfortable":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(150, 180),
            "dist_right": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "right_wall_near_comfortable":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_right": sample_in_range(150, 180),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "front_critical_left_clear":
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_NEAR, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "front_critical_right_clear":
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR),
            "dist_right": sample_in_range(OBSTACLE_NEAR, MAX_DIST),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "front_near_corridor":
        return {
            "dist_front": sample_in_range(OBSTACLE_CRITICAL + 10, OBSTACLE_NEAR),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(150, 200),
            "dist_right": sample_in_range(150, 200),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "front_near_open_sides":
        return {
            "dist_front": sample_in_range(OBSTACLE_CRITICAL + 10, OBSTACLE_NEAR),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_right": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "front_close_balanced":
        return {
            "dist_front": sample_in_range(OBSTACLE_NEAR + 50, OBSTACLE_CLOSE),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE - 50, OBSTACLE_CLOSE + 50),
            "dist_right": sample_in_range(OBSTACLE_CLOSE - 50, OBSTACLE_CLOSE + 50),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "dead_end":
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_NEAR),
            "dist_back": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "dist_left": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_right": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    elif scenario_type == "force_backward":
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_back": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_left": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_right": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_down": sample_in_range(VERTICAL_SAFE_DOWN, 700),
            "dist_up": sample_in_range(VERTICAL_SAFE_UP, MAX_DIST),
        }

    else:
        return {
            "dist_front": sample_in_range(MIN_DIST, MAX_DIST),
            "dist_back": sample_in_range(MIN_DIST, MAX_DIST),
            "dist_left": sample_in_range(MIN_DIST, MAX_DIST),
            "dist_right": sample_in_range(MIN_DIST, MAX_DIST),
            "dist_down": sample_in_range(100, MAX_DIST),
            "dist_up": sample_in_range(150, MAX_DIST),
        }


def validate_vertical_navigation(df):
    print("\n" + "=" * 70)
    print("VERTICAL NAVIGATION VALIDATION")
    print("=" * 70)

    ground_critical = df[df["dist_down"] < VERTICAL_CRITICAL_DOWN]
    if len(ground_critical) > 0:
        up_pct = (ground_critical["action"] == 7).mean() * 100.0
        print(f"Ground critical → UP: {up_pct:.1f}% (n={len(ground_critical)})")

    ceiling_critical = df[df["dist_up"] < VERTICAL_CRITICAL_UP]
    if len(ceiling_critical) > 0:
        down_pct = (ceiling_critical["action"] == 8).mean() * 100.0
        print(f"Ceiling critical → DOWN: {down_pct:.1f}% (n={len(ceiling_critical)})")

    ground_near = df[
        df["dist_down"].between(VERTICAL_CRITICAL_DOWN, VERTICAL_NEAR_DOWN)
        & (df["dist_up"] > VERTICAL_SAFE_UP)
    ]
    if len(ground_near) > 0:
        up_pct = (ground_near["action"] == 7).mean() * 100.0
        print(f"Ground near → UP: {up_pct:.1f}% (n={len(ground_near)})")

    print("=" * 70)


def validate_tight_space_navigation(df):
    print("\n" + "=" * 70)
    print("TIGHT SPACE VALIDATION")
    print("=" * 70)

    tight_corridor = df[
        (df["dist_front"] > SAFE_DISTANCE)
        & df["dist_left"].between(150, 200)
        & df["dist_right"].between(150, 200)
    ]

    if len(tight_corridor) > 0:
        forward_pct = (tight_corridor["action"] == 1).mean() * 100.0
        print(f"Tight corridor forward: {forward_pct:.1f}% (n={len(tight_corridor)})")

    print("=" * 70)


def generate_synthetic_data(num_samples):
    print("\nDRONE DATA GENERATOR - 6 FEATURES WITH VERTICAL AVOIDANCE\n")

    scenario_types = [
        ("ground_critical", 0.05),
        ("ceiling_critical", 0.04),
        ("ground_near", 0.05),
        ("ceiling_near", 0.03),
        ("vertical_squeeze", 0.02),
        ("ground_with_front_obstacle", 0.01),
        ("open_space", 0.04),
        ("tight_corridor", 0.18),
        ("very_tight_corridor", 0.12),
        ("left_wall_critical", 0.05),
        ("right_wall_critical", 0.05),
        ("left_wall_near_comfortable", 0.08),
        ("right_wall_near_comfortable", 0.08),
        ("front_critical_left_clear", 0.05),
        ("front_critical_right_clear", 0.05),
        ("front_near_corridor", 0.07),
        ("front_near_open_sides", 0.03),
        ("front_close_balanced", 0.02),
        ("dead_end", 0.02),
        ("force_backward", 0.01),
    ]

    data = []
    action_counts = {i: 0 for i in range(9)}

    min_per_action = 500
    forced_scenarios = {
        0: "open_space",
        7: "ground_critical",
        8: "ceiling_critical",
        1: "tight_corridor",
        2: "force_backward",
        3: "right_wall_critical",
        4: "left_wall_critical",
        5: "front_critical_left_clear",
        6: "front_critical_right_clear",
    }

    for action_id, scenario_type in forced_scenarios.items():
        attempts = 0
        generated = 0
        max_attempts = min_per_action * 30

        while generated < min_per_action and attempts < max_attempts:
            s = generate_scenario(scenario_type)
            a = apply_expert_rules(
                s["dist_front"],
                s["dist_back"],
                s["dist_left"],
                s["dist_right"],
                s["dist_down"],
                s["dist_up"],
            )
            if a == action_id:
                data.append(
                    {
                        "dist_front": s["dist_front"],
                        "dist_back": s["dist_back"],
                        "dist_left": s["dist_left"],
                        "dist_right": s["dist_right"],
                        "dist_down": s["dist_down"],
                        "dist_up": s["dist_up"],
                        "action": a,
                    }
                )
                action_counts[a] += 1
                generated += 1
            attempts += 1

        if generated < min_per_action:
            for _ in range(min_per_action - generated):
                s = generate_scenario(scenario_type)
                data.append(
                    {
                        "dist_front": s["dist_front"],
                        "dist_back": s["dist_back"],
                        "dist_left": s["dist_left"],
                        "dist_right": s["dist_right"],
                        "dist_down": s["dist_down"],
                        "dist_up": s["dist_up"],
                        "action": action_id,
                    }
                )
                action_counts[action_id] += 1

    remaining = num_samples - len(data)
    names = [s[0] for s in scenario_types]
    weights = [s[1] for s in scenario_types]

    for _ in range(remaining):
        stype = random.choices(names, weights=weights)[0]
        s = generate_scenario(stype)
        a = apply_expert_rules(
            s["dist_front"],
            s["dist_back"],
            s["dist_left"],
            s["dist_right"],
            s["dist_down"],
            s["dist_up"],
        )
        data.append(
            {
                "dist_front": s["dist_front"],
                "dist_back": s["dist_back"],
                "dist_left": s["dist_left"],
                "dist_right": s["dist_right"],
                "dist_down": s["dist_down"],
                "dist_up": s["dist_up"],
                "action": a,
            }
        )
        action_counts[a] += 1

    print(f"\n✅ Generated {len(data):,} samples\n")
    for aid in sorted(action_counts.keys()):
        c = action_counts[aid]
        pct = c / len(data) * 100.0
        print(f"  {aid} ({ACTION_MAP[aid]:10s}): {c:6d} ({pct:5.1f}%)")

    df = pd.DataFrame(data)
    validate_vertical_navigation(df)
    validate_tight_space_navigation(df)
    return df


def main():
    df = generate_synthetic_data(NUM_SAMPLES)
    output_file = "zigzag_drone_training_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    print(
        f"   Features: dist_front, dist_back, dist_left, dist_right, dist_down, dist_up"
    )
    print(f"   Actions: 0-8")


if __name__ == "__main__":
    main()
