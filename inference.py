"""
Inference Script — AI Disaster Response Coordinator
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    HF_TOKEN       Your Hugging Face API token (never commit the real value).
    API_BASE_URL   The API endpoint for the LLM (OpenAI-compatible Hugging Face router).
    MODEL_NAME     The model identifier to use for inference.
    IMAGE_NAME     The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=disaster_medium env=disaster_response model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action={"assignments":[{"vehicle_id":"veh_1","location_id":"loc_1"}]} reward=51.00 done=false error=null
    [END] success=true steps=5 rewards=51.00,21.00,14.00,12.00,8.00
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from models import DisasterAction, VehicleAssignment
    from client import DisasterResponseClient
except (ImportError, ModuleNotFoundError):
    from my_env import DisasterAction, DisasterResponseClient
    from my_env.models import VehicleAssignment

# ---------------------------------------------------------------------------
# Configuration — Hugging Face router via OpenAI-compatible client
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
if not HF_TOKEN:
    raise ValueError(
        "Set HF_TOKEN or OPENAI_API_KEY to your API token "
        "(create one at https://huggingface.co/settings/tokens — do not commit it)."
    )

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Docker image for environment container
IMAGE_NAME = os.getenv("IMAGE_NAME", "disaster_response-env:latest")

# Benchmark metadata
BENCHMARK = "disaster_response"

# Task configurations: name -> (difficulty, max_steps)
TASKS = {
    "disaster_easy": ("easy", 10),
    "disaster_medium": ("medium", 15),
    "disaster_hard": ("hard", 20),
}

# LLM parameters
TEMPERATURE = 0.3   # low temperature for deterministic decisions
MAX_TOKENS = 300

# ---------------------------------------------------------------------------
# System Prompt — instructs the LLM how to act in this environment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an AI Disaster Response Coordinator. Your goal is to rescue people from multiple locations using limited vehicles.

    ENVIRONMENT DYNAMICS:
    - Each vehicle can be assigned to exactly one location per step.
    - Vehicles rescue min(capacity, people_waiting) at the target location.
    - Locations with people left waiting incur increasing "waiting time" penalties.
    - Rewards: Rescue (+2.0), High Severity (+3.0 bonus), Medium Severity (+1.5 bonus).
    - Penalties: Idle vehicles (-1.0), Wasted dispatch (-2.0), Neglect (-0.5).

    STRATEGY (Follow these priorities):
    1. CAPACITY MATCHING: Match vehicle capacity to the number of people waiting. Don't waste a high-capacity vehicle on a site with few people if a smaller vehicle can handle it.
    2. SEVERITY-FIRST: Prioritize HIGH severity first, then MEDIUM, then LOW.
    3. WAIT-TIME MITIGATION: If multiple sites have the same severity, prioritize the one with the higher 'waiting_time' to minimize penalties.
    4. NO IDLE VEHICLES: Always assign every available vehicle to a site that still has people waiting.
    5. NO WASTED TRIPS: Never send a vehicle to a location where people_waiting is 0.

    OUTPUT FORMAT:
    You must provide your reasoning briefly, followed by the assignments in a JSON block code fence.

    Example Output:
    Reasoning: Location loc_1 has 20 people and is high priority. Assigning veh_1 (cap 10).
    ```json
    [{"vehicle_id": "veh_1", "location_id": "loc_1"}]
    ```
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str} score={score:.4f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation formatter — makes the state readable for the LLM
# ---------------------------------------------------------------------------


def format_observation_for_llm(obs: Any) -> str:
    """Convert a DisasterObservation into a clear text prompt for the LLM."""
    lines = []
    lines.append(f"TIME STEP: {obs.time_step} / {obs.max_steps}")
    lines.append(f"TOTAL PEOPLE SAVED SO FAR: {obs.total_people_saved}")
    lines.append(f"PEOPLE SAVED THIS STEP: {obs.people_saved_this_step}")
    lines.append("")

    # Locations
    lines.append("DISASTER LOCATIONS:")
    for loc in obs.locations:
        lines.append(
            f"  - {loc.name} (id={loc.id}): severity={loc.severity}, "
            f"people_waiting={loc.people_waiting}, waiting_time={loc.waiting_time}"
        )
    lines.append("")

    # Vehicles
    lines.append("RESCUE VEHICLES:")
    for v in obs.vehicles:
        lines.append(
            f"  - {v.name} (id={v.id}): capacity={v.capacity}, busy={v.is_busy}"
        )
    lines.append("")

    # Reward breakdown (if available)
    if obs.reward_breakdown:
        lines.append("LAST STEP REWARD BREAKDOWN:")
        for key, val in obs.reward_breakdown.items():
            lines.append(f"  {key}: {val:+.2f}")
        lines.append("")

    lines.append(
        "Assign each available vehicle to a location. "
        "Respond with ONLY a JSON array of assignments."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


import re

def parse_llm_assignments(text: str) -> List[Dict[str, str]]:
    """Parse the LLM response into a list of assignment dicts, supporting reasoning text."""
    # Try to find JSON inside markdown blocks first
    json_match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
    if not json_match:
        # Try finding any array-like structure
        json_match = re.search(r"(\[.*\])", text, re.DOTALL)

    if json_match:
        cleaned = json_match.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                valid = []
                for item in parsed:
                    if (
                        isinstance(item, dict)
                        and "vehicle_id" in item
                        and "location_id" in item
                    ):
                        valid.append({
                            "vehicle_id": str(item["vehicle_id"]),
                            "location_id": str(item["location_id"]),
                        })
                return valid
        except json.JSONDecodeError:
            pass

    return []


def get_model_assignments(
    client: OpenAI,
    obs_text: str,
    history: List[str],
) -> List[Dict[str, str]]:
    """Call the LLM to get vehicle assignments for this step."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include recent history for context (last 3 steps)
    if history:
        history_text = "\n".join(history[-3:])
        messages.append({"role": "user", "content": f"Previous steps:\n{history_text}"})
        messages.append({"role": "assistant", "content": "Understood, I'll use this context."})

    messages.append({"role": "user", "content": obs_text})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        assignments = parse_llm_assignments(text)
        if assignments:
            return assignments
        print(f"[DEBUG] Could not parse LLM response: {text!r}", flush=True)
        return []
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Grader — computes normalized score [0.0, 1.0] for each task
# ---------------------------------------------------------------------------


def compute_task_score(
    total_people_saved: int,
    total_people_initial: int,
    rewards: List[float],
    done_success: bool,
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    Compute a deterministic, normalized task score between 0.0 and 1.0.

    Scoring formula:
        - 60% weight: percentage of people rescued
        - 20% weight: completion bonus (1.0 if all rescued, 0.0 if timeout)
        - 20% weight: efficiency bonus (steps remaining / max_steps)
    """
    # Rescue percentage (0.0 to 1.0)
    rescue_pct = (
        total_people_saved / total_people_initial
        if total_people_initial > 0
        else 0.0
    )

    # Completion bonus
    completion_bonus = 1.0 if done_success else 0.0

    # Efficiency bonus: how many steps were saved
    efficiency = (
        (max_steps - steps_taken) / max_steps
        if max_steps > 0 and done_success
        else 0.0
    )

    score = (0.6 * rescue_pct) + (0.2 * completion_bonus) + (0.2 * efficiency)
    # Clamp score to strictly (0, 1) range as required by validator
    return min(max(score, 0.01), 0.99)


# Total people per difficulty (for grader)
TOTAL_PEOPLE = {
    "easy": 20,      # 15 + 5
    "medium": 53,    # 20 + 15 + 10 + 8
    "hard": 86,      # 25 + 18 + 12 + 15 + 10 + 6
}


# ---------------------------------------------------------------------------
# Run a single task episode
# ---------------------------------------------------------------------------


async def run_task(
    llm_client: OpenAI,
    env: DisasterResponseClient,
    task_name: str,
    difficulty: str,
    max_steps: int,
) -> float:
    """
    Run a single task episode and return the normalized score.
    """
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    total_people_initial = TOTAL_PEOPLE.get(difficulty, 53)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Set difficulty via environment variable before reset
        os.environ["DISASTER_DIFFICULTY"] = difficulty
        result = await env.reset()
        obs = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Format the observation for the LLM
            obs_text = format_observation_for_llm(obs)

            # Get assignments from the LLM
            raw_assignments = get_model_assignments(llm_client, obs_text, history)

            # Build the action
            action = DisasterAction(
                assignments=[
                    VehicleAssignment(
                        vehicle_id=a["vehicle_id"],
                        location_id=a["location_id"],
                    )
                    for a in raw_assignments
                ]
            )

            # Step the environment
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            # Log in the required format
            action_str = json.dumps(
                {"assignments": raw_assignments}, separators=(",", ":")
            )
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Build history entry for LLM context
            history.append(
                f"Step {step}: assigned {len(raw_assignments)} vehicles -> "
                f"rescued {obs.people_saved_this_step} people, reward={reward:+.2f}"
            )

            if done:
                break

        # Determine if episode ended with all people rescued
        all_rescued = all(loc.people_waiting == 0 for loc in obs.locations)

        # Compute normalized score using the grader
        score = compute_task_score(
            total_people_saved=obs.total_people_saved,
            total_people_initial=total_people_initial,
            rewards=rewards,
            done_success=all_rescued,
            steps_taken=steps_taken,
            max_steps=max_steps,
        )
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards, score=score)

    return score


# ---------------------------------------------------------------------------
# Main entry point — runs all 3 tasks
# ---------------------------------------------------------------------------


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Use the existing, running server at localhost:8000
    async with DisasterResponseClient(base_url="http://localhost:8000") as env:
        print("Connected to Local Environment. Starting evaluation...")
        scores = {}
        for task_name, (difficulty, max_steps) in TASKS.items():
            score = await run_task(llm_client, env, task_name, difficulty, max_steps)
            scores[task_name] = score
            print(f"[DEBUG] {task_name} score: {score:.3f}", flush=True)

        # Print final summary
        avg_score = sum(scores.values()) / len(scores)
        # Ensure average is also strictly within (0, 1)
        clamped_avg = min(max(avg_score, 0.01), 0.99)
        
        print(f"\n[SUMMARY] Average score: {clamped_avg:.4f}", flush=True)
        for name, sc in scores.items():
            print(f"  {name}: {sc:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())