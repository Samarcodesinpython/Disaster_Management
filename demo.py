"""
Full Demo: AI Disaster Response Coordinator
============================================
Runs a full episode on MEDIUM difficulty, showing the LLM agent
making rescue decisions step by step.

Uses the environment directly (no server needed) for a smooth demo.

Requires HF_TOKEN (your Hugging Face token — set in the environment, never in source).
Optional: API_BASE_URL, MODEL_NAME (defaults match Hugging Face router + Qwen).
"""
import json
import os
import sys

sys.path.insert(0, ".")

from openai import OpenAI
from server.my_env_environment import DisasterResponseEnvironment
from models import DisasterAction, VehicleAssignment

# ---------------------------------------------------------------------------
# Config (set via environment — do not commit secrets)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

SYSTEM_PROMPT = """You are an AI Disaster Response Coordinator managing rescue vehicles.
Assign vehicles to locations to rescue people. Prioritize HIGH severity locations.
Send higher-capacity vehicles to locations with more people waiting.
Never send a vehicle to a location with 0 people_waiting.
Always assign ALL available vehicles if there are tasks remaining.

Respond with ONLY a JSON array like: [{"vehicle_id": "veh_1", "location_id": "loc_1"}]
Do NOT include any explanation, just the JSON array."""


def print_reward_breakdown(breakdown):
    if not breakdown:
        return
    print("  REWARD BREAKDOWN:")
    for key, val in breakdown.items():
        if key == "termination_reason":
            reason = "SUCCESS" if val == 1.0 else "TIMEOUT"
            print(f"    termination: {reason}")
        elif val != 0:
            icon = "+" if val > 0 else "-"
            print(f"    [{icon}] {key}: {val:+.2f}")


def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print(
            "Set HF_TOKEN in your environment to your Hugging Face API token.",
            file=sys.stderr,
        )
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=hf_token)
    env = DisasterResponseEnvironment(difficulty="medium")

    print("=" * 60)
    print("  AI DISASTER RESPONSE COORDINATOR - LIVE DEMO")
    print("=" * 60)

    # Reset
    print("\n  Resetting environment...")
    obs = env.reset()

    total_people = sum(loc.people_waiting for loc in obs.locations)
    print(f"\n  SCENARIO: {len(obs.locations)} disaster locations, "
          f"{len(obs.vehicles)} rescue vehicles")
    print(f"  TOTAL PEOPLE TO RESCUE: {total_people}")
    print(f"  MAX STEPS: {obs.max_steps}")

    print("\n  INITIAL STATE:")
    for loc in obs.locations:
        sev = {"high": "[HIGH]", "medium": "[MED ]", "low": "[LOW ]"}.get(loc.severity, "")
        print(f"    {sev} {loc.name} ({loc.id}): {loc.people_waiting} people")
    print()
    for v in obs.vehicles:
        print(f"    VEHICLE {v.name} ({v.id}): capacity={v.capacity}")

    print("-" * 60)

    total_reward = 0.0
    done = obs.done

    for step in range(1, obs.max_steps + 1):
        if done:
            break

        # Build prompt for LLM
        prompt_lines = [
            f"TIME: {obs.time_step}/{obs.max_steps}, SAVED: {obs.total_people_saved}",
            "LOCATIONS:"
        ]
        for loc in obs.locations:
            prompt_lines.append(
                f"  {loc.id} ({loc.name}): severity={loc.severity}, "
                f"people_waiting={loc.people_waiting}, waiting_time={loc.waiting_time}"
            )
        prompt_lines.append("VEHICLES:")
        for v in obs.vehicles:
            prompt_lines.append(f"  {v.id} ({v.name}): capacity={v.capacity}")
        prompt_text = "\n".join(prompt_lines)

        print(f"\n  STEP {step}: Asking LLM for assignments...")

        # Get LLM response
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text + "\n\nAssign vehicles:"},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            raw = completion.choices[0].message.content.strip()
            print(f"  LLM Response: {raw}")

            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = cleaned[cleaned.index("\n") + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            assignments = json.loads(cleaned.strip())
        except Exception as e:
            print(f"  Parse error: {e}")
            assignments = []

        # Display assignments
        if assignments:
            print(f"  Assignments ({len(assignments)}):")
            for a in assignments:
                # Find location name
                loc_name = next(
                    (l.name for l in obs.locations if l.id == a.get("location_id", "")),
                    a.get("location_id", "?")
                )
                veh_name = next(
                    (v.name for v in obs.vehicles if v.id == a.get("vehicle_id", "")),
                    a.get("vehicle_id", "?")
                )
                print(f"     {veh_name} --> {loc_name}")
        else:
            print("  WARNING: No valid assignments!")

        # Build action and step environment
        action = DisasterAction(
            assignments=[
                VehicleAssignment(
                    vehicle_id=a["vehicle_id"],
                    location_id=a["location_id"]
                )
                for a in assignments
                if "vehicle_id" in a and "location_id" in a
            ]
        )

        obs = env.step(action)
        done = obs.done
        reward = obs.reward
        total_reward += reward

        print(f"\n  RESULTS:")
        print(f"     People rescued this step: {obs.people_saved_this_step}")
        print(f"     Total rescued so far:     {obs.total_people_saved} / {total_people}")
        print(f"     Step reward:              {reward:+.2f}")
        print(f"     Cumulative reward:        {total_reward:+.2f}")

        print_reward_breakdown(obs.reward_breakdown)

        # Show remaining locations
        remaining = [loc for loc in obs.locations if loc.people_waiting > 0]
        if remaining:
            print(f"\n  Locations still needing rescue:")
            for loc in remaining:
                sev = {"high": "[HIGH]", "medium": "[MED ]", "low": "[LOW ]"}.get(loc.severity, "")
                print(f"     {sev} {loc.name}: {loc.people_waiting} people, wait={loc.waiting_time}")
        else:
            print(f"\n  ALL LOCATIONS CLEAR!")

        print("-" * 60)

    # Final summary
    print("\n" + "=" * 60)
    if all(loc.people_waiting == 0 for loc in obs.locations):
        print("  MISSION SUCCESS! ALL PEOPLE RESCUED!")
    else:
        remaining_people = sum(loc.people_waiting for loc in obs.locations)
        print(f"  TIME'S UP! {remaining_people} people still waiting.")
    print(f"  Final Score: {obs.total_people_saved}/{total_people} rescued")
    print(f"  Total Reward: {total_reward:+.2f}")
    print(f"  Steps taken: {obs.time_step}")
    rescue_pct = obs.total_people_saved / total_people * 100 if total_people > 0 else 0
    print(f"  Rescue rate: {rescue_pct:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
