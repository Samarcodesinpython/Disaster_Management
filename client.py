# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AI Disaster Response Coordinator — Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        DisasterAction,
        DisasterObservation,
        LocationState,
        VehicleState,
    )
except (ImportError, ValueError):
    from models import (
        DisasterAction,
        DisasterObservation,
        LocationState,
        VehicleState,
    )


class DisasterResponseClient(
    EnvClient[DisasterAction, DisasterObservation, State]
):
    """
    Client for the AI Disaster Response Coordinator Environment.

    This client maintains a persistent WebSocket connection to the
    environment server, enabling efficient multi-step interactions
    with lower latency.  Each client instance has its own dedicated
    environment session on the server.

    Example:
        >>> with DisasterResponseClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.locations)
        ...
        ...     from models import VehicleAssignment
        ...     action = DisasterAction(assignments=[
        ...         VehicleAssignment(vehicle_id="veh_1", location_id="loc_1")
        ...     ])
        ...     result = client.step(action)
        ...     print(result.observation.total_people_saved)

    Example with Docker:
        >>> client = DisasterResponseClient.from_docker_image("disaster_response-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DisasterAction) -> Dict:
        """
        Convert DisasterAction to JSON payload for the step message.

        Args:
            action: DisasterAction instance with vehicle assignments.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "assignments": [
                {
                    "vehicle_id": a.vehicle_id,
                    "location_id": a.location_id,
                }
                for a in action.assignments
            ],
        }

    def _parse_result(self, payload: Dict) -> StepResult[DisasterObservation]:
        """
        Parse server response into StepResult[DisasterObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult containing the DisasterObservation.
        """
        obs_data = payload.get("observation", {})

        locations = [
            LocationState(**loc) for loc in obs_data.get("locations", [])
        ]
        vehicles = [
            VehicleState(**v) for v in obs_data.get("vehicles", [])
        ]

        observation = DisasterObservation(
            locations=locations,
            vehicles=vehicles,
            time_step=obs_data.get("time_step", 0),
            max_steps=obs_data.get("max_steps", 0),
            people_saved_this_step=obs_data.get("people_saved_this_step", 0),
            total_people_saved=obs_data.get("total_people_saved", 0),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
