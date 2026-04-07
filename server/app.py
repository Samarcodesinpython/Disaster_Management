# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
from fastapi.responses import HTMLResponse
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

# Disable Gradio UI (Web Interface)
os.environ["ENABLE_WEB_INTERFACE"] = "false"

app = create_app(
    DisasterResponseEnvironment,
    DisasterAction,
    DisasterObservation,
    env_name="disaster_response",
    max_concurrent_envs=1,
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>AI Disaster Response Coordinator</title>
            <style>
                body { font-family: sans-serif; text-align: center; padding: 50px; background-color: #f4f4f9; }
                h1 { color: #d32f2f; }
            </style>
        </head>
        <body>
            <h1>🚨 AI Disaster Response Coordinator</h1>
            <p><strong>OpenEnv Environment:</strong> Active and Running</p>
            <hr width="300">
            <p>Ready for evaluation at /reset and /step.</p>
        </body>
    </html>
    """


def main():
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    print(f"Starting disaster response server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
