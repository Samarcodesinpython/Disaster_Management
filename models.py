# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AI Disaster Response Coordinator Environment.

Defines the observation and action schemas for a disaster response simulation
where an AI agent assigns rescue vehicles to affected locations to maximize
lives saved and minimize response time.
"""

from typing import Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models (used inside Observation)
# ---------------------------------------------------------------------------


class LocationState(BaseModel):
    """State of a single disaster-affected location."""

    id: str = Field(..., description="Unique identifier for the location")
    name: str = Field(..., description="Human-readable name of the location")
    severity: str = Field(
        ..., description="Severity level: 'low', 'medium', or 'high'"
    )
    people_waiting: int = Field(
        ..., ge=0, description="Number of people still awaiting rescue"
    )
    waiting_time: int = Field(
        default=0,
        ge=0,
        description="Number of steps this location has been waiting without service",
    )


class VehicleState(BaseModel):
    """State of a single rescue vehicle."""

    id: str = Field(..., description="Unique identifier for the vehicle")
    name: str = Field(..., description="Human-readable name of the vehicle")
    capacity: int = Field(
        ...,
        gt=0,
        description="Maximum number of people this vehicle can rescue per step",
    )
    is_busy: bool = Field(
        default=False, description="Whether the vehicle is currently dispatched"
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class VehicleAssignment(BaseModel):
    """A single assignment mapping a vehicle to a location."""

    vehicle_id: str = Field(..., description="ID of the vehicle to dispatch")
    location_id: str = Field(..., description="ID of the target location")


class DisasterAction(Action):
    """
    Action for the Disaster Response environment.

    The agent outputs a list of vehicle-to-location assignments.
    Multiple assignments can be made in a single step.
    Invalid assignments are silently ignored.
    """

    assignments: List[VehicleAssignment] = Field(
        default_factory=list,
        description="List of vehicle-to-location assignments for this step",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class DisasterObservation(Observation):
    """
    Observation from the Disaster Response environment.

    Contains the full state of all locations, vehicles, and timing info.
    Designed to be easily parsed and reasoned about by an LLM agent.
    """

    locations: List[LocationState] = Field(
        default_factory=list,
        description="Current state of all disaster-affected locations",
    )
    vehicles: List[VehicleState] = Field(
        default_factory=list,
        description="Current state of all rescue vehicles",
    )
    time_step: int = Field(
        default=0, description="Current time step in the episode"
    )
    max_steps: int = Field(
        default=0, description="Maximum steps before episode timeout"
    )
    people_saved_this_step: int = Field(
        default=0, description="Number of people rescued in the last step"
    )
    total_people_saved: int = Field(
        default=0, description="Cumulative people rescued so far in the episode"
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed breakdown of the reward components for explainability",
    )
