# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AI Disaster Response Coordinator — Environment Implementation.

Simulates a disaster scenario where an AI agent assigns limited rescue
vehicles to multiple affected locations in order to maximise lives saved
and minimise response time.

This is NOT an RL training system.  The environment only:
  - provides state (observation)
  - accepts actions (vehicle assignments)
  - returns reward, done flag, and updated state

The agent is external (LLM-based inference script).
"""

import copy
import os
from typing import Any, Dict, List, Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        DisasterAction,
        DisasterObservation,
        LocationState,
        VehicleState,
    )
    from ..tasks import get_scenario
except ImportError:
    from models import (
        DisasterAction,
        DisasterObservation,
        LocationState,
        VehicleState,
    )
    from tasks import get_scenario


class DisasterResponseEnvironment(Environment):
    """
    AI Disaster Response Coordinator Environment.

    The agent must assign rescue vehicles to disaster-affected locations
    each step.  The goal is to maximise lives saved while prioritising
    high-severity locations and minimising response delays.

    ------------------------------------------------------------------
    Reward Design (per step)
    ------------------------------------------------------------------

    Positive rewards:
        +2.0  per person rescued
        +3.0  bonus per person rescued from HIGH severity location
        +1.5  bonus per person rescued from MEDIUM severity location
        +1.0  per served location with waiting_time <= 2 (quick response)

    Negative penalties:
        -0.5  per unserved location that still has people waiting
        -0.3  per unit of waiting time added to neglected locations
        -2.0  per vehicle dispatched to a location with 0 people
        -1.0  per idle vehicle when rescue tasks are still pending

    ------------------------------------------------------------------
    Done Condition
    ------------------------------------------------------------------
        - All people across all locations are rescued   → done (success)
        - Maximum number of steps reached               → done (timeout)

    ------------------------------------------------------------------
    Dispatch Model
    ------------------------------------------------------------------
    Single-step dispatch: vehicles complete their assignment within the
    step and are available again in the next step.  This keeps the action
    space simple for an LLM agent.
    """

    # Allow multiple WebSocket clients, each getting their own instance.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = None):
        """
        Initialise the environment.

        Args:
            difficulty: Scenario difficulty ("easy", "medium", "hard").
                        Falls back to env var DISASTER_DIFFICULTY, then "medium".
        """
        self._difficulty: str = (
            difficulty or os.getenv("DISASTER_DIFFICULTY", "medium")
        )
        self._state: State = State(episode_id=str(uuid4()), step_count=0)

        # Internal mutable state — populated properly on reset()
        self._locations: List[Dict[str, Any]] = []
        self._vehicles: List[Dict[str, Any]] = []
        self._time_step: int = 0
        self._max_steps: int = 0
        self._total_people_saved: int = 0

    # ==================================================================
    # Public API
    # ==================================================================

    def reset(self, *, seed: int = None, difficulty: str = None, **kwargs) -> DisasterObservation:
        """
        Reset the environment to a fresh scenario.

        Loads the scenario for the configured difficulty, initialises all
        internal state, and returns the first observation.

        Returns:
            DisasterObservation with the initial scenario state.
        """
        scenario = get_scenario(self._difficulty)

        # Deep-copy so each episode starts from a clean slate
        self._locations = copy.deepcopy(scenario.locations)
        self._vehicles = copy.deepcopy(scenario.vehicles)
        self._max_steps = scenario.max_steps
        self._time_step = 0
        self._total_people_saved = 0

        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._build_observation(
            reward=0.0,
            done=False,
            reward_breakdown={},
            people_saved_this_step=0,
        )

    def step(self, action: DisasterAction) -> DisasterObservation:  # type: ignore[override]
        """
        Execute one environment step.

        Process flow:
            1. Validate & deduplicate assignments
            2. Rescue people at assigned locations
            3. Increment waiting_time for unserved locations
            4. Compute reward
            5. Advance time_step
            6. Check termination condition
            7. Free vehicles for the next step
            8. Return observation

        Args:
            action: DisasterAction containing vehicle-to-location assignments.

        Returns:
            DisasterObservation with updated state, reward, and done flag.
        """
        # 1. Validate and deduplicate assignments
        valid_assignments = self._validate_assignments(action.assignments)

        # 2. Process valid assignments — rescue people
        step_stats = self._process_assignments(valid_assignments)

        # 3. Update waiting times for unserved locations
        self._update_waiting_times(step_stats["served_location_ids"])

        # 4. Gather additional statistics needed for reward
        step_stats.update(
            self._gather_reward_stats(
                step_stats["served_location_ids"],
                step_stats["assigned_vehicle_ids"],
            )
        )

        # 5. Compute reward
        reward, breakdown = self._compute_reward(step_stats)

        # 6. Advance time step
        self._time_step += 1
        self._state.step_count = self._time_step

        # 7. Check termination
        all_rescued = all(
            loc["people_waiting"] == 0 for loc in self._locations
        )
        timeout = self._time_step >= self._max_steps
        done = all_rescued or timeout

        if done:
            breakdown["termination_reason"] = (
                1.0 if all_rescued else 0.0  # 1.0 = success, 0.0 = timeout
            )

        # 8. Reset vehicles for next step (single-step dispatch model)
        for vehicle in self._vehicles:
            vehicle["is_busy"] = False

        # 9. Build and return observation
        return self._build_observation(
            reward=reward,
            done=done,
            reward_breakdown=breakdown,
            people_saved_this_step=step_stats["total_rescued"],
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # ==================================================================
    # Internal: Validation
    # ==================================================================

    def _validate_assignments(
        self, assignments: list
    ) -> List[Dict[str, str]]:
        """
        Validate and deduplicate vehicle assignments.

        Rules applied (in order):
            - vehicle_id must reference an existing vehicle
            - location_id must reference an existing location
            - each vehicle can only be assigned once per step (first wins)
            - all invalid or duplicate assignments are silently skipped

        Returns:
            List of validated assignment dicts {vehicle_id, location_id}.
        """
        valid_vehicle_ids: Set[str] = {v["id"] for v in self._vehicles}
        valid_location_ids: Set[str] = {loc["id"] for loc in self._locations}
        seen_vehicles: Set[str] = set()
        valid: List[Dict[str, str]] = []

        for assignment in assignments:
            vid = assignment.vehicle_id
            lid = assignment.location_id

            # Skip if vehicle or location doesn't exist
            if vid not in valid_vehicle_ids or lid not in valid_location_ids:
                continue

            # Skip duplicate vehicle assignments (first assignment wins)
            if vid in seen_vehicles:
                continue

            seen_vehicles.add(vid)
            valid.append({"vehicle_id": vid, "location_id": lid})

        return valid

    # ==================================================================
    # Internal: Processing
    # ==================================================================

    def _process_assignments(
        self, valid_assignments: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Process validated assignments: rescue people, update vehicle status.

        For each valid assignment:
            - rescued = min(vehicle.capacity, location.people_waiting)
            - location.people_waiting is reduced by rescued amount
            - vehicle is marked as busy

        Returns:
            Dict of step statistics used for reward computation.
        """
        total_rescued: int = 0
        high_severity_rescued: int = 0
        medium_severity_rescued: int = 0
        wasted_dispatches: int = 0
        quick_response_count: int = 0
        served_location_ids: Set[str] = set()
        assigned_vehicle_ids: Set[str] = set()

        # Build lookup maps for O(1) access
        vehicle_map: Dict[str, Dict] = {v["id"]: v for v in self._vehicles}
        location_map: Dict[str, Dict] = {
            loc["id"]: loc for loc in self._locations
        }

        for assignment in valid_assignments:
            vid = assignment["vehicle_id"]
            lid = assignment["location_id"]
            vehicle = vehicle_map[vid]
            location = location_map[lid]

            assigned_vehicle_ids.add(vid)

            # Wasted dispatch: no one left to rescue at this location
            if location["people_waiting"] == 0:
                wasted_dispatches += 1
                vehicle["is_busy"] = True
                continue

            # Rescue: take min(capacity, people_waiting)
            rescued = min(vehicle["capacity"], location["people_waiting"])
            location["people_waiting"] -= rescued
            vehicle["is_busy"] = True

            total_rescued += rescued
            self._total_people_saved += rescued
            served_location_ids.add(lid)

            # Track severity for bonus rewards
            severity = location["severity"]
            if severity == "high":
                high_severity_rescued += rescued
            elif severity == "medium":
                medium_severity_rescued += rescued

            # Quick response bonus: served while waiting_time <= 2
            if location["waiting_time"] <= 2:
                quick_response_count += 1

        return {
            "total_rescued": total_rescued,
            "high_severity_rescued": high_severity_rescued,
            "medium_severity_rescued": medium_severity_rescued,
            "wasted_dispatches": wasted_dispatches,
            "quick_response_count": quick_response_count,
            "served_location_ids": served_location_ids,
            "assigned_vehicle_ids": assigned_vehicle_ids,
        }

    def _update_waiting_times(self, served_location_ids: Set[str]) -> None:
        """
        Increment waiting_time for every unserved location that still
        has people waiting.  This reflects the cost of being neglected.
        """
        for location in self._locations:
            if (
                location["id"] not in served_location_ids
                and location["people_waiting"] > 0
            ):
                location["waiting_time"] += 1

    def _gather_reward_stats(
        self,
        served_location_ids: Set[str],
        assigned_vehicle_ids: Set[str],
    ) -> Dict[str, int]:
        """
        Compute additional metrics needed for the reward function.

        Returns:
            Dict with unserved_with_people, waiting_time_increase,
            and idle_with_pending counts.
        """
        # Count unserved locations that still have people waiting
        unserved_with_people: int = sum(
            1
            for loc in self._locations
            if loc["id"] not in served_location_ids
            and loc["people_waiting"] > 0
        )

        # Total waiting time increase equals # of neglected locations
        waiting_time_increase: int = unserved_with_people

        # Count idle vehicles while rescue tasks remain
        has_pending: bool = any(
            loc["people_waiting"] > 0 for loc in self._locations
        )
        idle_with_pending: int = 0
        if has_pending:
            idle_with_pending = sum(
                1
                for v in self._vehicles
                if v["id"] not in assigned_vehicle_ids
            )

        return {
            "unserved_with_people": unserved_with_people,
            "waiting_time_increase": waiting_time_increase,
            "idle_with_pending": idle_with_pending,
        }

    # ==================================================================
    # Internal: Reward
    # ==================================================================

    def _compute_reward(
        self, stats: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the step reward based on rescue outcomes.

        The reward is designed to reflect real-world disaster response
        priorities: saving lives is paramount, high-severity locations
        are urgent, and delays have compounding negative consequences.

        Positive components:
            rescue_reward           = total_rescued          × 2.0
            high_severity_bonus     = high_severity_rescued  × 3.0
            medium_severity_bonus   = medium_severity_rescued× 1.5
            quick_response_bonus    = quick_response_count   × 1.0

        Negative components:
            unserved_location_penalty = unserved_with_people × 0.5
            waiting_time_penalty      = waiting_time_increase× 0.3
            wasted_dispatch_penalty   = wasted_dispatches    × 2.0
            idle_vehicle_penalty      = idle_with_pending    × 1.0

        Returns:
            Tuple of (total_reward, breakdown_dict).
        """
        reward: float = 0.0
        breakdown: Dict[str, float] = {}

        # --- POSITIVE REWARDS ---

        # +2.0 per person rescued (core incentive)
        rescue_reward = stats["total_rescued"] * 2.0
        reward += rescue_reward
        breakdown["rescue_reward"] = rescue_reward

        # +3.0 bonus per person from HIGH severity locations
        high_bonus = stats["high_severity_rescued"] * 3.0
        reward += high_bonus
        breakdown["high_severity_bonus"] = high_bonus

        # +1.5 bonus per person from MEDIUM severity locations
        med_bonus = stats["medium_severity_rescued"] * 1.5
        reward += med_bonus
        breakdown["medium_severity_bonus"] = med_bonus

        # +1.0 per location served quickly (waiting_time <= 2)
        quick_bonus = stats["quick_response_count"] * 1.0
        reward += quick_bonus
        breakdown["quick_response_bonus"] = quick_bonus

        # --- NEGATIVE PENALTIES ---

        # -0.5 per unserved location that still has people
        unserved_pen = stats["unserved_with_people"] * 0.5
        reward -= unserved_pen
        breakdown["unserved_location_penalty"] = -unserved_pen

        # -0.3 per unit of waiting time added to neglected locations
        waiting_pen = stats["waiting_time_increase"] * 0.3
        reward -= waiting_pen
        breakdown["waiting_time_penalty"] = -waiting_pen

        # -2.0 per vehicle dispatched to a location with 0 people
        wasted_pen = stats["wasted_dispatches"] * 2.0
        reward -= wasted_pen
        breakdown["wasted_dispatch_penalty"] = -wasted_pen

        # -1.0 per idle vehicle when rescue tasks remain
        idle_pen = stats["idle_with_pending"] * 1.0
        reward -= idle_pen
        breakdown["idle_vehicle_penalty"] = -idle_pen

        breakdown["total_reward"] = reward
        return reward, breakdown

    # ==================================================================
    # Internal: Observation Builder
    # ==================================================================

    def _build_observation(
        self,
        reward: float,
        done: bool,
        reward_breakdown: Dict[str, float],
        people_saved_this_step: int,
    ) -> DisasterObservation:
        """
        Build a DisasterObservation from the current internal state.

        Converts internal dicts into typed Pydantic models for clean
        JSON serialisation over the wire.
        """
        locations = [
            LocationState(
                id=loc["id"],
                name=loc["name"],
                severity=loc["severity"],
                people_waiting=loc["people_waiting"],
                waiting_time=loc["waiting_time"],
            )
            for loc in self._locations
        ]

        vehicles = [
            VehicleState(
                id=v["id"],
                name=v["name"],
                capacity=v["capacity"],
                is_busy=v["is_busy"],
            )
            for v in self._vehicles
        ]

        return DisasterObservation(
            locations=locations,
            vehicles=vehicles,
            time_step=self._time_step,
            max_steps=self._max_steps,
            people_saved_this_step=people_saved_this_step,
            total_people_saved=self._total_people_saved,
            reward_breakdown=reward_breakdown,
            done=done,
            reward=reward,
        )
