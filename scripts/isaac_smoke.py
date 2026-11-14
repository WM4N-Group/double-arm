"""SERVER ONLY: bounded headless Isaac Sim startup, not a task success test."""
import argparse
import ctypes
import json
from pathlib import Path
import os
import sys
import traceback

if os.environ.get("ACCEPT_NVIDIA_EULA") != "YES":
    sys.exit("Explicit NVIDIA EULA acceptance required (ACCEPT_NVIDIA_EULA=YES).")
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

parser = argparse.ArgumentParser()
parser.add_argument('--render-output', type=Path)
parser.add_argument('--result-file', type=Path)
args = parser.parse_args()
# Fail early on a missing loader dependency rather than accepting an app warning.
ctypes.CDLL('libGLU.so.1')
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
failed = False
try:
    for _ in range(30):
        app.update()
    report = {"isaac_startup": "passed", "updates": 30, "task_success": "not_tested"}
    if args.render_output:
        import numpy as np
        import omni.replicator.core as rep
        from PIL import Image
        with rep.new_layer():
            material = rep.create.material_omnipbr(diffuse=(0.15, 0.5, 0.9))
            cube = rep.create.cube(scale=(0.01, 0.01, 0.01))
            with cube:
                rep.randomizer.materials(material)
            rep.create.light(light_type='Dome', intensity=1000)
            camera = rep.create.camera(position=(3, 3, 3), look_at=(0, 0, 0))
            product = rep.create.render_product(camera, (320, 240))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([product])
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=4)
        image = np.asarray(rgb.get_data())
        if image.shape[:2] != (240, 320) or image.size == 0 or float(image.std()) < 1.0:
            raise RuntimeError(f'No usable rendered image: shape={image.shape}')
        args.render_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(args.render_output)
        report.update(render_test='passed', image=str(args.render_output), image_shape=list(image.shape))
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
except BaseException:
    failed = True
    traceback.print_exc()
finally:
    try:
        app.close()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            failed = True
if failed:
    raise SystemExit(1)
