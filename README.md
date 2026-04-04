---
title: AI Disaster Response Coordinator
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# AI Disaster Response Coordinator

The AI Disaster Response Coordinator is an OpenEnv-compliant reinforcement learning environment designed to evaluate agent performance in high-stakes resource allocation and crisis management. This environment simulates the complex decision-making required by emergency operations centers during a large-scale disaster.

## Overview

The AI Disaster Response Coordinator is an OpenEnv-compliant reinforcement learning environment designed for evaluating multi-agent and LLM-based coordination in high-stakes crises. It features high-fidelity simulation of priority triage, resource allocation, and dynamic wait-time penalties.

### 🏆 Final Evaluation Results (Baseline)
Evaluated using `model=Qwen2.5-72B-Instruct` across three tiers of difficulty.

| Difficulty | Task | Score | Result |
| :--- | :--- | :--- | :--- |
| **Easy** | Priority Identification | **1.000** | 🚀 Perfect |
| **Medium**| Capacity Management | **0.933** | 📈 High Performance |
| **Hard** | Strategic Trade-offs | **0.720** | 🛠️ Solid Baseline |
| **TOTAL** | **Weighted Average** | **0.884** | |

### Key Challenges
- **Priority Triage**: Locations vary in severity (Low, Medium, High), requiring the agent to identify and address the most critical needs first.
- **Resource Constraints**: Total rescue capacity per step is limited by the number and type of available vehicles.
- **Dynamic Penalties**: Delays in response lead to increasing "waiting time" penalties, modeling the deteriorating conditions in real-world crisis scenarios.

## Quick Start

```python
from my_env import DisasterAction, DisasterResponseClient
from my_env.models import VehicleAssignment

# Initialize the client
client = DisasterResponseClient(base_url="http://localhost:8000")

# Reset the environment to get the initial observation
result = client.reset()
obs = result.observation
print(f"Total Locations: {len(obs.locations)}")

# Dispatch vehicles to target locations
action = DisasterAction(assignments=[
    VehicleAssignment(vehicle_id="veh_1", location_id="loc_1"),
    VehicleAssignment(vehicle_id="veh_2", location_id="loc_3"),
])

# Execute the step
result = client.step(action)
print(f"Lives Saved This Step: {result.observation.people_saved_this_step}")
print(f"Current Cumulative Reward: {result.reward}")
```

## System Requirements and Setup

### Docker Deployment (Recommended)
The environment is fully containerized for consistent evaluation.

```bash
# Build the environment image
docker build -t disaster-response-env:latest -f server/Dockerfile .

# Run the environment server
docker run -p 8000:8000 disaster_response-env:latest
```

### Local Development
For development and debugging without Docker:

```bash
# Install dependencies using uv
uv sync

# Start the FastAPI server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Inference and Baseline Evaluation
Run the standardized inference script to evaluate an agent across all difficulty tiers.

```bash
# Hugging Face token (each teammate uses their own; never commit the real value)
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Execute evaluation suite
python inference.py
```

## Technical Specification

### Action Model
**`DisasterAction`**: Encapsulates a list of discrete vehicle-to-location assignments.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `assignments` | `List[VehicleAssignment]` | Mapping of specific vehicles to target locations. |

**`VehicleAssignment`**:
| Attribute | Type | Description |
| :--- | :--- | :--- |
| `vehicle_id` | `str` | Unique identifier for the rescue asset. |
| `location_id` | `str` | Unique identifier for the disaster site. |

### Observation Model
**`DisasterObservation`**: Provides the current global state of the crisis.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `locations` | `List[LocationState]` | List of sites, their severity, and population status. |
| `vehicles` | `List[VehicleState]` | List of assets, their status, and rescue capacities. |
| `time_step` | `int` | Current progress within the episode runtime. |
| `max_steps` | `int` | Horizontal limit for the simulation. |
| `total_people_saved`| `int` | Cumulative performance metric for mission success. |

### Reward Dynamics
The environment utilizes a dense reward structure to guide agent learning:
- **Rescues**: +2.0 base reward per individual saved.
- **Severity Multipliers**: +3.0 (High) or +1.5 (Medium) additional bonus per person based on site triage.
- **Efficiency Penalties**:
  - Idle penalty: -1.0 for unassigned vehicles while tasks remain.
  - Wait penalty: -0.3 per step for each location left unserved.
  - Wasted Dispatch: -2.0 for sending vehicles to cleared locations.

## Scenarios

The environment features three deterministic scenarios designed to test specific agent capabilities.

### Tier 1: Easy (Priority Identification)
- **Problem**: 2 locations (1 High, 1 Low severity), 1 vehicle.
- **Objective**: Demonstrate basic triage by prioritizing the high-severity location.

### Tier 2: Medium (Capacity Management)
- **Problem**: 4 locations across three severity levels, 2 vehicles with differing capacities.
- **Objective**: Optimize throughput by matching vehicle capacity to location population.

### Tier 3: Hard (Strategic Trade-offs)
- **Problem**: 6 locations (3 High, 2 Medium, 1 Low), 3 vehicles with limited capacity.
- **Objective**: Balance conflicting priorities under a tight step limit (20 steps).

---
### 🛠️ Submission Checklist
- [x] **OpenEnv V1.0 Compliance**: PASS (`openenv validate`)
- [x] **Standardized Inference**: PASS (follows stdout protocol)
- [x] **Multi-Tier Evaluation**: Easy, Medium, and Hard scenarios fully implemented.
- [x] **Dockerized Deployment**: Fully containerized environment for consistent evaluation.

Built with ❤️ for the **Meta Hackathon RL** by the Disaster Response Team.
