# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os

import uvicorn

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with 'uv sync'") from e

try:
    from ..models import DisasterAction, DisasterObservation
    from .my_env_environment import DisasterResponseEnvironment
except (ImportError, ModuleNotFoundError):
    from models import DisasterAction, DisasterObservation
    from server.my_env_environment import DisasterResponseEnvironment

app = create_app(
    DisasterResponseEnvironment,
    DisasterAction,
    DisasterObservation,
    env_name="disaster_response",
    max_concurrent_envs=1,
)


def main():
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    print(f"Starting disaster response server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
