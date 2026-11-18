#!/usr/bin/env python3
"""Remote CPU-only X7 CAD import, joint-drive verification and rendered review."""
import argparse
from bisect import bisect_right
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p

HZ, FPS, W, H, PANEL = 240, 24, 960, 540, 320
QUAD = ("#5dd9f7", "#f7aa61")
SIM_POSE = ([0, 0, 1.77], [0, -math.pi / 2, 0])
NEUTRAL = np.array([0, .45, 0, -.87, 0, .33, 0])
KEYS = [
    (0, NEUTRAL, NEUTRAL),
    (1.5, NEUTRAL, NEUTRAL),
    (4.0, [.56, .78, -.42, -1.25, .4, .55, .66], NEUTRAL),
    (5.5, NEUTRAL, NEUTRAL),
    (8.0, NEUTRAL, [-.56, .78, .42, -1.25, -.4, .55, -.66]),
    (10.5, [-.45, .25, .48, -.54, -.45, .55, -.62], [.45, .25, -.48, -.54, .45, .55, .62]),
    (13.0, [.3, .68, -.26, -1.15, .31, .15, .46], [-.3, .68, .26, -1.15, -.31, .15, -.46]),
    (15.5, NEUTRAL, NEUTRAL),
    (17.0, NEUTRAL, NEUTRAL),
]


def cluster_stl(stl, spacing):
    """Cluster visual vertices and remove collapsed or duplicate triangles."""
    dtype = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    verts = np.frombuffer(stl, dtype=dtype, offset=84)["v"].reshape(-1, 3)
    cell = np.rint(verts / spacing).astype(np.int32)
    _, inv = np.unique(cell, axis=0, return_inverse=True)
    number = np.bincount(inv)
    centroids = np.column_stack([np.bincount(inv, weights=verts[:, k]) / number for k in range(3)])
    faces = inv.reshape(-1, 3)
    mask = ((faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2]))
    faces = faces[mask]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique)]
    out = np.zeros(len(faces), dtype=dtype)
    out["v"] = centroids[faces]
    normals = np.cross(out["v"][:, 1] - out["v"][:, 0], out["v"][:, 2] - out["v"][:, 0])
    norms = np.linalg.norm(normals, axis=1)
    good = norms > 1e-12
    out["normal"][good] = normals[good] / norms[good, None]
    return stl[:80] + int(good.sum()).to_bytes(4, "little") + out[good].tobytes()


def stage_asset(source, directory):
    """Make a derivative URDF; never rewrite the vendor ZIP or meshes."""
    directory.mkdir()
    meshes = directory / "meshes"
    meshes.mkdir()
    counts = {}
    with ZipFile(source) as z:
        [urdf] = [name for name in z.namelist() if name.endswith(".urdf")]
        root = ET.fromstring(z.read(urdf))
        archive_dir = PurePosixPath(urdf).parent.parent
        for link in root.findall("link"):
            name = link.attrib["name"]
            visual = link.find("visual")
            mesh = visual.find("geometry/mesh")
            filename = PurePosixPath(mesh.attrib["filename"]).name
            stl = z.read(str(archive_dir / "meshes" / filename))
            if len(stl) == 80 and name in ("Link7", "Link14"):
                link.remove(visual)
                link.remove(link.find("collision"))
                counts[name] = [0, 0]
                continue
            count = int.from_bytes(stl[80:84], "little")
            if len(stl) != 84 + 50 * count or count == 0:
                raise ValueError(f"Invalid STL: {filename}")
            if name != "base_link":
                collision = link.find("collision/geometry/mesh")
                (meshes / ("collision_" + filename)).write_bytes(stl)
                collision.set("filename", "meshes/collision_" + filename)
            else:
                # The fixed root's detailed CAD mesh is visual-only in this test.
                link.remove(link.find("collision"))
            stl = cluster_stl(stl, .020 if name == "base_link" else .006)
            counts[name] = [count, int.from_bytes(stl[80:84], "little")]
            (meshes / filename).write_bytes(stl)
            mesh.set("filename", "meshes/" + filename)
    urdf_path = directory / "x7_duo_fixed_base.urdf"
    ET.ElementTree(root).write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path, counts


def setup(urdf):
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / HZ)
    p.setPhysicsEngineParameter(numSolverIterations=80, deterministicOverlappingPairs=1)
    robot = p.loadURDF(str(urdf), SIM_POSE[0], p.getQuaternionFromEuler(SIM_POSE[1]),
                       useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
    indices = {p.getJointInfo(robot, i)[1].decode(): i for i in range(p.getNumJoints(robot))}
    if len(indices) != 14:
        raise ValueError(f"Wrong joint count: {indices}")
    joints = [indices[f"Joint{i}"] for i in range(1, 15)]
    for k, i in enumerate(joints):
        if p.getJointInfo(robot, i)[2] != p.JOINT_REVOLUTE:
            raise ValueError(f"Joint{k + 1} not revolute")
        # Export has two coincident joints at several housings. The neighboring
        # mesh convex hulls overlap by design; use non-neighboring pairs only.
        for l, j in enumerate(joints[:k]):
            if k // 7 == l // 7 and k - l <= 2:
                p.setCollisionFilterPair(robot, robot, i, j, 0)
        shade = (0.27, 0.67, .8, 1) if k < 7 else (.84, .43, .19, 1)
        if k % 2:
            shade = tuple(.24 + v*.57 for v in shade[:3]) + (1,)
        p.changeVisualShape(robot, i, rgbaColor=shade)
    p.changeVisualShape(robot, -1, rgbaColor=(.73, .79, .85, 1))
    for i, q in zip(joints, np.tile(NEUTRAL, 2)):
        p.resetJointState(robot, i, q)
    forces = np.array([p.getJointInfo(robot, i)[10] for i in joints])
    return robot, joints, forces


def target(t):
    j = min(bisect_right([k[0] for k in KEYS], t) - 1, len(KEYS) - 2)
    t0, a0, b0 = KEYS[j]
    t1, a1, b1 = KEYS[j+1]
    u = (t - t0) / (t1 - t0)
    s = u**3 * (10 + u * (-15 + 6*u))
    q0, q1 = np.r_[a0, b0], np.r_[a1, b1]
    return (1 - s) * q0 + s * q1, (30 * u**2 * (1-u)**2 / (t1-t0)) * (q1-q0)


def decorate_world():
    def box(extents, xyz, rgba):
        shape = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=xyz)
    box([6, 6, .012], [0, 0, -.032], [.15, .20, .27, 1])
    for i in range(-5, 6):
        c = [.215, .278, .345, 1]
        box([.010, 2.5, .001], [i * .50, 0, -.018], c)
        box([2.5, .010, .001], [0, i * .50, -.018], c)


def make_renderer():
    base = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    title, medium, body, small = (ImageFont.truetype(base + s + ".ttf", n) for s, n in
                                   [("-Bold", 31), ("-Bold", 17), ("", 16), ("", 13)])
    projection = p.computeProjectionMatrixFOV(fov=39, aspect=W/H, nearVal=.05, farVal=25)

    def render(t, q, desired, error):
        yaw = 15 + 15 * math.cos(t / KEYS[-1][0] * 2 * math.pi)
        view = p.computeViewMatrixFromYawPitchRoll([0, -.055, .94], 4.30, yaw, -9, 0, 2)
        _, _, rgba, _, seg = p.getCameraImage(W, H, viewMatrix=view, projectionMatrix=projection,
                                          renderer=p.ER_TINY_RENDERER, shadow=0,
                                          lightDirection=[1, -3, 5], lightDistance=5,
                                          lightAmbientCoeff=.57, lightDiffuseCoeff=.52, lightSpecularCoeff=.18)
        img = Image.new("RGB", (W + PANEL, H), "#111b28")
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3].copy()
        rgb[np.asarray(seg).reshape(H, W) < 0] = (213, 225, 238)
        img.paste(Image.fromarray(rgb), (0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((22, 18, 320, 101), 14, fill="#111b28")
        d.text((40, 27), "X7 DUO", font=title, fill="#f3f7fb")
        d.text((41, 71), "FIXED BASE  /  DUAL 7-DOF  /  PYBULLET", font=small, fill="#b7c7d8")
        d.text((W+22, 20), "MEASURED JOINT ANGLES", font=medium, fill="#f3f7fb")
        d.text((W+22, 46), "Actual position  /  |  target", font=small, fill="#9fb2c8")
        for group, color in enumerate(QUAD):
            y0 = 79 + group*186
            d.text((W+21, y0), f"ARM {'AB'[group]}   /   J{1+7*group}–J{7+7*group}", font=medium, fill=color)
            for k in range(7):
                i = 7*group + k
                y = y0 + 34 + k*20
                d.text((W+22, y-8), f"J{i+1:02}", font=small, fill="#b7c7d8")
                center = W + 147
                d.line((W+66, y, W+228, y), fill="#33475b", width=4)
                d.line((center, y-4, center, y+4), fill="#8a9baa", width=1)
                pos, ref = [center + 52 * float(np.clip(v, -1.55, 1.55)) for v in (q[i], desired[i])]
                d.line((min(center, pos), y, max(center, pos), y), fill=color, width=4)
                d.line((ref, y-5, ref, y+5), fill="#ffffff", width=1)
                d.text((W+238, y-8), f"{math.degrees(q[i]):+5.1f}°", font=small, fill="#e7eff8")
        phase = ("HOLD INITIAL POSE" if t <= 1.5 else "ARM A ONLY" if t <= 5.5 else
                 "ARM B ONLY" if t <= 8 else "COORDINATED DUAL-ARM MOTION" if t <= 15.5 else "HOLD FINAL POSE")
        d.rectangle((0, H-59, W+PANEL, H), fill="#111b28")
        d.text((23, H-46), phase, font=body, fill="#eef5ff")
        d.text((W+22, H-46), f"Max joint error  {math.degrees(error):.2f}°", font=body, fill="#b7c7d8")
        d.text((W-106, H-44), f"{t:04.1f} s", font=small, fill="#b7c7d8")
        d.line((24, H-16, W+PANEL-24, H-16), fill="#354b5e", width=2)
        d.line((24, H-16, 24+(W+PANEL-48)*t/KEYS[-1][0], H-16), fill=QUAD[0], width=2)
        return img
    return render


def simulate(robot, joints, forces, out, render_video):
    base = np.array(p.getBasePositionAndOrientation(robot)[0])
    track, samples, speeds = [], [], []
    contacts = {}
    peak_effort = np.zeros(14)
    base_drift = 0.0
    video = None
    if render_video:
        decorate_world()
        render = make_renderer()
        video = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-threads", "1",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{W+PANEL}x{H}",
            "-framerate", str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-threads", "1",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out / "x7-duo-fixed-base.mp4"),
        ], stdin=subprocess.PIPE)
    steps = int(KEYS[-1][0] * HZ)
    try:
        for k in range(1, steps + 1):
            t = k / HZ
            q_des, dq_des = target(t)
            p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL,
                                        targetPositions=q_des, targetVelocities=dq_des,
                                        forces=forces, positionGains=[.23]*14, velocityGains=[1.0]*14)
            p.stepSimulation()
            states = p.getJointStates(robot, joints)
            q = np.array([s[0] for s in states])
            dq = np.array([s[1] for s in states])
            effort = np.array([s[3] for s in states])
            if not np.isfinite(np.r_[q, dq, effort]).all():
                raise RuntimeError(f"Non-finite physics state at step {k}")
            error = np.abs(q - q_des)
            track.append(error)
            samples.append(q)
            speeds.append(abs(dq))
            peak_effort = np.maximum(peak_effort, abs(effort))
            base_drift = max(base_drift, np.linalg.norm(np.array(p.getBasePositionAndOrientation(robot)[0]) - base))
            for contact in p.getContactPoints(robot, robot):
                a, b = contact[3:5]
                key = tuple(sorted((a+1, b+1)))
                row = contacts.setdefault(key, [0, 0.0])
                row[0] += 1
                row[1] = max(row[1], float(contact[9]))
            if video and k % (HZ // FPS) == 0:
                img = render(t, q, q_des, error.max())
                video.stdin.write(img.tobytes())
                if k in {HZ, 4*HZ, 8*HZ, int(10.5*HZ), 13*HZ, 16*HZ}:
                    img.save(out / f"view-{t:04.1f}s.jpg", quality=91)
    finally:
        if video:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("Video encoding failed")
    angles, errors = np.array(samples), np.array(track)
    return {
        "steps": steps, "simulated_seconds": steps / HZ, "physics_hz": HZ,
        "video_frames": steps // (HZ // FPS) if video else 0,
        "fixed_base_translation_max_m": base_drift,
        "tracking_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(errors**2)))),
        "tracking_peak_deg": float(np.rad2deg(errors.max())),
        "tracking_p95_deg": float(np.rad2deg(np.percentile(errors, 95))),
        "motion_range_per_joint_deg": np.rad2deg(np.ptp(angles, axis=0)).tolist(),
        "peak_speed_per_joint_rad_s": np.max(speeds, axis=0).tolist(),
        "peak_applied_motor_effort_per_joint_Nm": peak_effort.tolist(),
        "enabled_self_contact_pairs_point_count_and_peak_force_N": {f"{a}-{b}": v for (a, b), v in contacts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("This remote CPU workflow requires CUDA_VISIBLE_DEVICES='' explicitly")
    args.out.mkdir(parents=True)
    urdf, counts = stage_asset(args.source, args.out / "derived-model")
    print(f"Prepared input mesh triangle counts: {counts}", flush=True)
    robot, joints, forces = setup(urdf)
    try:
        report = simulate(robot, joints, forces, args.out, not args.no_video)
    finally:
        p.disconnect()
    passed = (min(report["motion_range_per_joint_deg"]) > 5
              and report["tracking_peak_deg"] < 3
              and report["fixed_base_translation_max_m"] < 1e-8
              and not report["enabled_self_contact_pairs_point_count_and_peak_force_N"])
    report |= {"joint_names": [f"Joint{i}" for i in range(1, 15)], "joint_smoke_pass": passed,
               "simulator": f"PyBullet {version('pybullet')} DIRECT, CPU, TinyRenderer",
               "source_zip_sha256": sha256(args.source.read_bytes()).hexdigest(), "mesh_triangle_counts": counts,
               "asset_changes": ["Rotated CAD frame to upright; fixed root; visual mesh clustering: root 20 mm, arms 6 mm",
                                 "Root and empty Link7/Link14 collisions omitted; adjacent/next-adjacent same-arm pairs ignored",
                                 "No gripper, object contact, or hardware calibration validated"]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
