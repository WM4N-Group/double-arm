#!/usr/bin/env python3
"""SERVER ONLY: same recorded policy actions, assisted / physical grasp audit.

This is counterfactual action replay for diagnosis, NOT learned-policy success.
No resulting video or task status may be used as a policy acceptance result.
"""
import argparse
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from grasping_validation import activate_physical_after_reset, require_empty_assisted_grasps
from state_policy_features import as_numpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--demo',default='demo_1');p.add_argument('--max-steps',type=int,default=1100)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    for name in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS'):
        if os.environ.get(name)!='YES':raise RuntimeError('Previously recorded explicit consent required')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES';torch.set_num_threads(4);torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True)
    manifest=json.loads((args.policy_run/'manifest.json').read_text())
    actions=np.load(args.policy_run/f'evaluation/{args.demo}-trace.npz')['actions'][:args.max_steps]
    with h5py.File(manifest['dataset']) as file:initial=file[f'data/{args.demo}/states'][0]
    meta=manifest['env_args'];meta['env_kwargs']['init_curobo']=False
    import omnigibson as og
    from omnigibson.macros import macros
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    report={'mode':'counterfactual_recorded_policy_action_grasp_audit','learned_policy_evaluation':False,
            'task_success_claimed':False,'policy_run':str(args.policy_run),'demo':args.demo,'cases':[]}
    failed=False
    try:
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs'])
        robot=env.robot;original_mode=robot.grasping_mode
        cup=env.env.task.object_scope['teacup.n.02_1'].unwrapped
        original_set_pose=cup.set_position_orientation
        for mode in ('configured','physical'):
            cup.set_position_orientation=original_set_pose
            for arm in robot.arm_names:
                if robot._ag_obj_constraints[arm] is not None:robot.release_grasp_immediately(arm)
            robot._grasping_mode=original_mode;env.env.reset()
            if og.sim.dump_state(serialized=True).numel()!=len(initial):raise ValueError('Serialized reset layout mismatch')
            og.sim.load_state(torch.tensor(initial,dtype=torch.float32),serialized=True)
            require_empty_assisted_grasps(robot)
            if mode=='physical':activate_physical_after_reset(robot)
            case={'requested_mode':mode,'actual_mode':robot.grasping_mode,'initial_state_load_count':1,
                  'grasp_window_s':float(macros.robots.manipulation_robot.GRASP_WINDOW),'pose_write_calls':[],
                  'first_assisted_constraint_step':None,'steps':len(actions)}
            step=-1
            def tracked_set_pose(*a,**kw):
                case['pose_write_calls'].append({'step':step,'keys':list(kw)})
                return original_set_pose(*a,**kw)
            cup.set_position_orientation=tracked_set_pose
            samples=[];start=time.time()
            with (args.output/f'{mode}-contact-window.jsonl').open('w',buffering=1) as log:
                for step,action in enumerate(actions):
                    env.env.step(torch.tensor(action,dtype=torch.float32))
                    pos,quat=cup.get_position_orientation();eef_pos,eef_quat=robot.eef_links['left'].get_position_orientation()
                    ag=any(v is not None for v in robot._ag_obj_constraints.values())
                    if mode=='physical' and ag:raise RuntimeError('Assisted constraint appeared in physical test')
                    if ag and case['first_assisted_constraint_step'] is None:case['first_assisted_constraint_step']=step
                    samples.append(np.concatenate([as_numpy(pos),as_numpy(quat),as_numpy(eef_pos),as_numpy(eef_quat),[int(ag)]]))
                    if 750<=step<=820:
                        _,contacts=robot._find_gripper_contacts(arm='left')
                        record={'step':step,'cup_position_world_m':as_numpy(pos).tolist(),'cup_quaternion_xyzw':as_numpy(quat).tolist(),
                                'eef_position_world_m':as_numpy(eef_pos).tolist(),'assisted_constraint_active':ag,
                                'cup_contacting_robot_links':sorted(contacts.get(cup.root_link.prim_path,set()))}
                        log.write(json.dumps(record)+'\n')
                    if step%200==0:print(json.dumps({'mode':mode,'step':step,'assisted_constraint_active':ag}),flush=True)
            cup.set_position_orientation=original_set_pose
            values=np.stack(samples);np.savez_compressed(args.output/f'{mode}-poses.npz',poses=values,actions=actions)
            case.update(elapsed_s=time.time()-start,pose_write_count=len(case['pose_write_calls']),
                        final_cup_eef_distance_m=float(np.linalg.norm(values[-1,:3]-values[-1,7:10])),
                        cup_displacement_since_start_m=float(np.linalg.norm(values[-1,:3]-values[0,:3])))
            report['cases'].append(case)
            (args.output/'audit-result.json').write_text(json.dumps(report,indent=2))
    except BaseException as error:
        import traceback
        failed=True;report['error']=repr(error);traceback.print_exc()
    finally:
        (args.output/'audit-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
        try:og.shutdown()
        except SystemExit:pass
    if failed:raise SystemExit(7)


if __name__=='__main__':main()
