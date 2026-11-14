"""SERVER ONLY: bounded headless Isaac Sim startup, not a task success test."""
import json
import os
import sys

if os.environ.get("ACCEPT_NVIDIA_EULA") != "YES":
    sys.exit("Explicit NVIDIA EULA acceptance required (ACCEPT_NVIDIA_EULA=YES).")
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "width": 640,
        "height": 480,
        "active_gpu": 0,
        "physics_gpu": 0,
        "multi_gpu": False,
    }
)
try:
    for _ in range(30):
        app.update()
    print(json.dumps({"isaac_startup": "passed", "updates": 30, "task_success": "not_tested"}))
finally:
    app.close()
