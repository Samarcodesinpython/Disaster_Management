# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task Configurations for the AI Disaster Response Coordinator.

Defines three difficulty levels (easy, medium, hard) with deterministic
scenario initialisation. Each scenario specifies disaster locations,
rescue vehicles, and episode constraints.

All scenarios are fully deterministic — no randomness involved.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScenarioConfig:
    """
    Immutable configuration for a single disaster-response scenario.

    Attributes:
        difficulty: Human-readable difficulty label.
        locations: List of location dicts (id, name, severity, people_waiting, waiting_time).
        vehicles:  List of vehicle  dicts (id, name, capacity, is_busy).
        max_steps: Maximum number of steps before the episode times out.
    """

    difficulty: str
    locations: List[Dict]
    vehicles: List[Dict]
    max_steps: int


# ---------------------------------------------------------------------------
# Easy — Obvious decision-making
# ---------------------------------------------------------------------------


def get_easy_scenario() -> ScenarioConfig:
    """
    Easy scenario: 2 locations, 1 vehicle, 10 max steps.

    One high-severity location (15 people) and one low-severity (5 people).
    The optimal strategy is obvious: serve the high-severity site first,
    then mop up the low-severity site.
    """
    return ScenarioConfig(
        difficulty="easy",
        locations=[
            {
                "id": "loc_1",
                "name": "Central Hospital",
                "severity": "high",
                "people_waiting": 15,
                "waiting_time": 0,
            },
            {
                "id": "loc_2",
                "name": "Park Shelter",
                "severity": "low",
                "people_waiting": 5,
                "waiting_time": 0,
            },
        ],
        vehicles=[
            {
                "id": "veh_1",
                "name": "Rescue Truck Alpha",
                "capacity": 10,
                "is_busy": False,
            },
        ],
        max_steps=10,
    )


# ---------------------------------------------------------------------------
# Medium — Requires prioritisation and capacity planning
# ---------------------------------------------------------------------------


def get_medium_scenario() -> ScenarioConfig:
    """
    Medium scenario: 4 locations, 2 vehicles, 15 max steps.

    Mixed severities with different people counts.
    Two vehicles have different capacities, forcing the agent to think
    about which vehicle goes where for maximum impact.
    """
    return ScenarioConfig(
        difficulty="medium",
        locations=[
            {
                "id": "loc_1",
                "name": "Downtown Hospital",
                "severity": "high",
                "people_waiting": 20,
                "waiting_time": 0,
            },
            {
                "id": "loc_2",
                "name": "Riverside School",
                "severity": "medium",
                "people_waiting": 15,
                "waiting_time": 0,
            },
            {
                "id": "loc_3",
                "name": "Industrial Zone",
                "severity": "high",
                "people_waiting": 10,
                "waiting_time": 0,
            },
            {
                "id": "loc_4",
                "name": "Suburban Mall",
                "severity": "low",
                "people_waiting": 8,
                "waiting_time": 0,
            },
        ],
        vehicles=[
            {
                "id": "veh_1",
                "name": "Rescue Truck Alpha",
                "capacity": 8,
                "is_busy": False,
            },
            {
                "id": "veh_2",
                "name": "Rescue Truck Beta",
                "capacity": 5,
                "is_busy": False,
            },
        ],
        max_steps=15,
    )


# ---------------------------------------------------------------------------
# Hard — Trade-offs, time pressure, conflicting priorities
# ---------------------------------------------------------------------------


def get_hard_scenario() -> ScenarioConfig:
    """
    Hard scenario: 6 locations, 3 vehicles, 20 max steps.

    Three high-severity locations compete for limited vehicle capacity.
    The agent must make genuine trade-off decisions every single step.
    Total people (86) vs total capacity per step (18) means ~5 perfect
    steps minimum, but priorities and penalties make it much harder.
    """
    return ScenarioConfig(
        difficulty="hard",
        locations=[
            {
                "id": "loc_1",
                "name": "City Hospital",
                "severity": "high",
                "people_waiting": 25,
                "waiting_time": 0,
            },
            {
                "id": "loc_2",
                "name": "Collapsed Bridge",
                "severity": "high",
                "people_waiting": 18,
                "waiting_time": 0,
            },
            {
                "id": "loc_3",
                "name": "Flooded School",
                "severity": "medium",
                "people_waiting": 12,
                "waiting_time": 0,
            },
            {
                "id": "loc_4",
                "name": "Chemical Plant",
                "severity": "high",
                "people_waiting": 15,
                "waiting_time": 0,
            },
            {
                "id": "loc_5",
                "name": "Residential Block",
                "severity": "medium",
                "people_waiting": 10,
                "waiting_time": 0,
            },
            {
                "id": "loc_6",
                "name": "Community Center",
                "severity": "low",
                "people_waiting": 6,
                "waiting_time": 0,
            },
        ],
        vehicles=[
            {
                "id": "veh_1",
                "name": "Rescue Truck Alpha",
                "capacity": 8,
                "is_busy": False,
            },
            {
                "id": "veh_2",
                "name": "Rescue Truck Beta",
                "capacity": 6,
                "is_busy": False,
            },
            {
                "id": "veh_3",
                "name": "Rescue Truck Gamma",
                "capacity": 4,
                "is_busy": False,
            },
        ],
        max_steps=20,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

SCENARIOS = {
    "easy": get_easy_scenario,
    "medium": get_medium_scenario,
    "hard": get_hard_scenario,
}


def get_scenario(difficulty: str = "medium") -> ScenarioConfig:
    """
    Return a ScenarioConfig for the requested difficulty level.

    Args:
        difficulty: One of "easy", "medium", "hard". Case-insensitive.

    Returns:
        A deterministic ScenarioConfig for the requested difficulty.

    Raises:
        ValueError: If difficulty is not one of the valid options.
    """
    difficulty = difficulty.lower().strip()
    if difficulty not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS.keys()))
        raise ValueError(
            f"Invalid difficulty '{difficulty}'. Must be one of: {valid}"
        )
    return SCENARIOS[difficulty]()
