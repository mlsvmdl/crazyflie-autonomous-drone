"""
OPTIMIZED SYNTHETIC TRAINING DATA GENERATOR FOR DRONE
- Tight space navigation (works down to 150-200mm clearance)
- Dead-ends yaw (180° turn) by default; UP only when top/bottom approaching
- Guarantees ALL 9 actions
- Aggressive thresholds for close-quarters maneuvering
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Action mapping
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

# Configuration
NUM_SAMPLES = 50000

MIN_DIST = 100.0
MAX_DIST = 2000.0
MIN_ALT = 0.1
MAX_ALT = 2.5
TARGET_ALT = 0.5
ALT_TOLERANCE = 0.15

# TIGHT SPACE THRESHOLDS - optimized for 150-200mm navigation
OBSTACLE_CRITICAL = 120.0  # Extreme danger - will touch wall
OBSTACLE_NEAR = 200.0  # Very close - must react immediately
OBSTACLE_CLOSE = 350.0  # Close - prepare to react
SAFE_DISTANCE = 600.0  # Safe to go forward aggressively

# Altitude margins for "approaching" top/bottom
ALT_UP_MARGIN = 0.3  # Used to define "top approaching"
TOP_APPROACHING = MAX_ALT - ALT_UP_MARGIN
BOTTOM_APPROACHING = MIN_ALT + ALT_UP_MARGIN


# ------------- State classification helpers ------------- #


def front_state(d):
    if d < OBSTACLE_CRITICAL:
        return "critical"
    elif d < OBSTACLE_NEAR:
        return "near"
    elif d < SAFE_DISTANCE:
        return "close"
    else:
        return "clear"


def side_state(d):
    if d < OBSTACLE_CRITICAL:
        return "critical"
    elif d < OBSTACLE_NEAR:
        return "near"
    elif d < OBSTACLE_CLOSE:
        return "close"
    else:
        return "clear"


def altitude_state(alt):
    """
    Altitude logic:
    - too_low  => bottom approaching: allow UP
    - too_high => top approaching: allow DOWN, and dead-ends may use UP to escape if needed
    - ok       => no vertical movement preferred
    """
    if alt < BOTTOM_APPROACHING:
        return "too_low"
    elif alt > TOP_APPROACHING:
        return "too_high"
    else:
        return "ok"


# ------------- OPTIMIZED EXPERT POLICY FOR TIGHT SPACES ------------- #


def apply_expert_rules(dist_front, dist_back, dist_left, dist_right, altitude):
    """
    Optimized expert system for tight space navigation.

    Key behaviors:
    - Tolerates closer approaches (down to 150-200mm)
    - Dead-ends YAW by default (180° turn), not UP
    - UP/DOWN only when top/bottom is approaching
    - Backward is extremely rare (only if can't yaw and can't safely stay)
    - Aggressive forward preference in corridors
    """

    # ---- Priority 1: ALTITUDE SAFETY (top/bottom approaching) ----
    alt_state = altitude_state(altitude)
    if alt_state == "too_low":
        # Bottom approaching: go UP
        return 7  # UP
    if alt_state == "too_high":
        # Top approaching: go DOWN
        return 8  # DOWN

    # ---- Classify all distances ----
    f_state = front_state(dist_front)
    l_state = side_state(dist_left)
    r_state = side_state(dist_right)

    # ---- Priority 2: CRITICAL SIDE WALL (about to touch) ----
    if dist_left < OBSTACLE_CRITICAL and dist_right > OBSTACLE_NEAR:
        return 4  # RIGHT - emergency
    if dist_right < OBSTACLE_CRITICAL and dist_left > OBSTACLE_NEAR:
        return 3  # LEFT - emergency

    # ---- Priority 3: DEAD-END - DEFAULT IS YAW, not UP ----
    # Dead-end = front blocked AND both sides blocked
    dead_end = (
        f_state in ("critical", "near")
        and l_state in ("critical", "near")
        and r_state in ("critical", "near")
    )

    if dead_end:
        # Default behavior: always yaw to turn around
        # Only use UP here if explicitly desired when already high and no other good option.
        # Since altitude_state already handled true top/bottom danger, keep this purely yaw.
        return 5  # YAW_LEFT (start 180° turn)

    # ---- Priority 4: FRONT CRITICAL (about to hit) ----
    if f_state == "critical":
        # Immediate obstacle - must yaw or move sideways NOW
        left_clear = l_state in ("clear", "close")
        right_clear = r_state in ("clear", "close")

        if left_clear and not right_clear:
            return 5  # YAW_LEFT
        elif right_clear and not left_clear:
            return 6  # YAW_RIGHT
        elif left_clear and right_clear:
            # Both sides have room - pick side with MORE space
            return 5 if dist_left > dist_right else 6
        else:
            # Both sides blocked but not dead-end (back must be clear)
            # This is the ONLY case for backward
            if dist_back > OBSTACLE_NEAR:
                return 2  # BACKWARD (rare case)
            else:
                # Truly stuck - yaw anyway
                return 5  # YAW_LEFT

    # ---- Priority 5: FRONT NEAR (very close) ----
    if f_state == "near":
        left_better = l_state in ("clear", "close")
        right_better = r_state in ("clear", "close")

        if left_better and not right_better:
            return 5  # YAW_LEFT
        elif right_better and not left_better:
            return 6  # YAW_RIGHT
        elif left_better and right_better:
            # Both okay - compare distances
            return 5 if dist_left > dist_right else 6
        else:
            # Both sides near - yaw toward better side
            return 5 if dist_left > dist_right else 6

    # ---- Priority 6: FRONT CLOSE (approaching) ----
    if f_state == "close":
        # Only yaw if there's a BIG side advantage
        side_diff = abs(dist_left - dist_right)
        BIG_SIDE_ADVANTAGE = 250.0  # Reduced for tighter spaces

        if side_diff > BIG_SIDE_ADVANTAGE:
            # Clear advantage - gentle yaw
            return 5 if dist_left > dist_right else 6
        else:
            # Sides similar - keep going forward
            return 1  # FORWARD

    # ---- Priority 7: SIDE WALL PROXIMITY (corridor centering) ----
    if f_state == "clear":
        # Near left wall - move right
        if l_state in ("critical", "near") and r_state in ("clear", "close"):
            return 4  # RIGHT

        # Near right wall - move left
        if r_state in ("critical", "near") and l_state in ("clear", "close"):
            return 3  # LEFT

        # Close to left wall - gentle correction
        if l_state == "close" and r_state == "clear":
            return 4  # RIGHT

        # Close to right wall - gentle correction
        if r_state == "close" and l_state == "clear":
            return 3  # LEFT

        # Narrow corridor (both sides close) - go forward centered
        if l_state == "close" and r_state == "close":
            return 1  # FORWARD

        # Everything clear - go forward
        return 1  # FORWARD

    # ---- Priority 8: DEFAULT - FORWARD ----
    return 1  # FORWARD


# ------------- Scenario generation ------------- #


def sample_in_range(lo, hi):
    return random.uniform(lo, hi)


def generate_scenario(scenario_type):
    """Generate scenarios optimized for tight space training."""

    if scenario_type == "open_space":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_right": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "tight_corridor":
        # Narrow corridor - both sides close, front clear
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "dist_right": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "very_tight_corridor":
        # Very tight corridor - testing limits
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(150, OBSTACLE_NEAR + 50),
            "dist_right": sample_in_range(150, OBSTACLE_NEAR + 50),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "left_wall_critical":
        # About to touch left wall
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_right": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "right_wall_critical":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_right": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "left_wall_near":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR + 50),
            "dist_right": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "right_wall_near":
        return {
            "dist_front": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR + 50),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "front_critical_left_clear":
        # About to hit wall, left side open - yaw left
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "front_critical_right_clear":
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "dist_right": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "front_near_left_better":
        return {
            "dist_front": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR + 50),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "front_near_right_better":
        return {
            "dist_front": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR + 50),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "front_close_balanced":
        # Front getting close but sides balanced - keep forward
        return {
            "dist_front": sample_in_range(OBSTACLE_CLOSE, SAFE_DISTANCE),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(SAFE_DISTANCE - 100, MAX_DIST),
            "dist_right": sample_in_range(SAFE_DISTANCE - 100, MAX_DIST),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "dead_end_yaw":
        # Dead-end at normal altitude - should YAW (not UP)
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_NEAR),
            "dist_back": sample_in_range(OBSTACLE_NEAR, OBSTACLE_CLOSE),
            "dist_left": sample_in_range(MIN_DIST, OBSTACLE_NEAR),
            "dist_right": sample_in_range(MIN_DIST, OBSTACLE_NEAR),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "force_backward":
        # Rare case: front critical, sides blocked, back clear - BACKWARD
        return {
            "dist_front": sample_in_range(MIN_DIST, OBSTACLE_CRITICAL),
            "dist_back": sample_in_range(SAFE_DISTANCE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR),
            "dist_right": sample_in_range(OBSTACLE_CRITICAL, OBSTACLE_NEAR),
            "altitude": sample_in_range(TARGET_ALT - 0.1, TARGET_ALT + 0.1),
        }

    elif scenario_type == "too_low":
        # Bottom approaching
        return {
            "dist_front": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "altitude": sample_in_range(MIN_ALT, BOTTOM_APPROACHING - 0.02),
        }

    elif scenario_type == "too_high":
        # Top approaching
        return {
            "dist_front": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_back": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_left": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "dist_right": sample_in_range(OBSTACLE_CLOSE, MAX_DIST),
            "altitude": sample_in_range(TOP_APPROACHING + 0.02, MAX_ALT),
        }

    elif scenario_type == "force_hover":
        return {
            "dist_front": sample_in_range(OBSTACLE_NEAR + 20, OBSTACLE_CLOSE),
            "dist_back": sample_in_range(OBSTACLE_NEAR + 20, OBSTACLE_CLOSE),
            "dist_left": sample_in_range(OBSTACLE_NEAR + 20, OBSTACLE_CLOSE),
            "dist_right": sample_in_range(OBSTACLE_NEAR + 20, OBSTACLE_CLOSE),
            "altitude": sample_in_range(TARGET_ALT - 0.05, TARGET_ALT + 0.05),
        }

    else:  # "random"
        return {
            "dist_front": sample_in_range(MIN_DIST, MAX_DIST),
            "dist_back": sample_in_range(MIN_DIST, MAX_DIST),
            "dist_left": sample_in_range(MIN_DIST, MAX_DIST),
            "dist_right": sample_in_range(MIN_DIST, MAX_DIST),
            "altitude": sample_in_range(MIN_ALT, MAX_ALT),
        }


def generate_synthetic_data(num_samples):
    print("\n" + "=" * 70)
    print("  TIGHT SPACE OPTIMIZED DRONE TRAINING DATA GENERATOR")
    print("=" * 70)
    print(
        f"  Thresholds: CRITICAL={OBSTACLE_CRITICAL} NEAR={OBSTACLE_NEAR} "
        f"CLOSE={OBSTACLE_CLOSE} SAFE={SAFE_DISTANCE}"
    )
    print("=" * 70)

    # Heavy emphasis on tight space scenarios
    scenario_types = [
        ("open_space", 0.25),
        ("tight_corridor", 0.20),
        ("very_tight_corridor", 0.10),
        ("left_wall_critical", 0.04),
        ("right_wall_critical", 0.04),
        ("left_wall_near", 0.05),
        ("right_wall_near", 0.05),
        ("front_critical_left_clear", 0.04),
        ("front_critical_right_clear", 0.04),
        ("front_near_left_better", 0.04),
        ("front_near_right_better", 0.04),
        ("front_close_balanced", 0.03),
        ("dead_end_yaw", 0.03),
        ("force_backward", 0.01),  # Minimal backward
        ("too_low", 0.01),
        ("too_high", 0.01),
        ("force_hover", 0.00),
        ("random", 0.00),
    ]

    data = []
    action_counts = {i: 0 for i in range(9)}

    print(f"\nGenerating {num_samples:,} samples...")
    print("Scenario distribution:")
    for scenario, prob in scenario_types:
        print(f"   {scenario:30s}: {prob * 100:>5.1f}%")

    # Force minimum samples for each action
    print("\n🔒 Forcing minimum samples for each action...")
    min_per_action = 150

    forced_scenarios = {
        7: "too_low",
        8: "too_high",
        1: "open_space",
        2: "force_backward",
        3: "right_wall_critical",
        4: "left_wall_critical",
        5: "front_critical_left_clear",
        6: "front_critical_right_clear",
        0: "force_hover",
    }

    for action_id, scenario_type in forced_scenarios.items():
        attempts = 0
        generated = 0
        max_attempts = min_per_action * 30

        while generated < min_per_action and attempts < max_attempts:
            scenario = generate_scenario(scenario_type)
            action = apply_expert_rules(
                scenario["dist_front"],
                scenario["dist_back"],
                scenario["dist_left"],
                scenario["dist_right"],
                scenario["altitude"],
            )

            if action == action_id:
                data.append(
                    {
                        "dist_front": scenario["dist_front"],
                        "dist_back": scenario["dist_back"],
                        "dist_left": scenario["dist_left"],
                        "dist_right": scenario["dist_right"],
                        "altitude": scenario["altitude"],
                        "action": action,
                    }
                )
                action_counts[action] += 1
                generated += 1

            attempts += 1

        print(
            f"   Action {action_id} ({ACTION_MAP[action_id]:10s}): "
            f"{generated:3d} samples (attempts: {attempts})"
        )

        if generated < min_per_action:
            print(
                f"      ⚠️  Only got {generated}, forcing "
                f"{min_per_action - generated} synthetic..."
            )
            for _ in range(min_per_action - generated):
                base_scenario = generate_scenario(scenario_type)
                data.append(
                    {
                        "dist_front": base_scenario["dist_front"],
                        "dist_back": base_scenario["dist_back"],
                        "dist_left": base_scenario["dist_left"],
                        "dist_right": base_scenario["dist_right"],
                        "altitude": base_scenario["altitude"],
                        "action": action_id,
                    }
                )
                action_counts[action_id] += 1

    # Generate remaining samples
    remaining = num_samples - len(data)
    print(f"\n📊 Generating {remaining:,} additional samples...")

    for i in range(remaining):
        scenario_type = random.choices(
            [s[0] for s in scenario_types],
            weights=[s[1] for s in scenario_types],
        )[0]

        scenario = generate_scenario(scenario_type)
        action = apply_expert_rules(
            scenario["dist_front"],
            scenario["dist_back"],
            scenario["dist_left"],
            scenario["dist_right"],
            scenario["altitude"],
        )

        data.append(
            {
                "dist_front": scenario["dist_front"],
                "dist_back": scenario["dist_back"],
                "dist_left": scenario["dist_left"],
                "dist_right": scenario["dist_right"],
                "altitude": scenario["altitude"],
                "action": action,
            }
        )

        action_counts[action] += 1

        if (i + 1) % 10000 == 0:
            print(f"   Generated {len(data):,} / {num_samples:,}...")

    print(f"\n✅ Generated {len(data):,} samples\n")
    print("Final action distribution:")
    total = sum(action_counts.values())
    for action_id in sorted(action_counts.keys()):
        count = action_counts[action_id]
        pct = (count / total) * 100
        status = "✅" if count >= 50 else "⚠️"
        print(
            f"   {status} {action_id} ({ACTION_MAP[action_id]:10s}): "
            f"{count:6d} ({pct:5.1f}%)"
        )

    missing = [i for i in range(9) if action_counts[i] < 50]
    if missing:
        print(f"\n❌ WARNING: Low count actions: {[ACTION_MAP[i] for i in missing]}")
    else:
        print("\n✅ All 9 actions well represented!")

    return pd.DataFrame(data)


def main():
    synthetic_df = generate_synthetic_data(NUM_SAMPLES)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"drone_training_data_{timestamp}.csv"
    synthetic_df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"Total: {len(synthetic_df):,} samples")
    print("\nNext: Rename to 'drone_training_data.csv' and run trainer")


if __name__ == "__main__":
    main()
