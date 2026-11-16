#!/usr/bin/env python3
"""SERVER ONLY diagnostic: replay recorded ACTIONS from one initial state.

This is NOT a learned policy or a state-replay video. It tests whether the
saved actions remain executable with the teacher planner removed. Never use
its outcome to claim successful policy training or publish a policy demo.
"""
import argparse
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from state_policy_features import as_numpy
from object_pose_audit import ObjectPoseAudit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--demo',default='demo_1')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reset-settle-steps',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.reset_settle_steps<=20:raise ValueError('Invalid reset settling')
    for key in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS'):
        if os.environ.get(key)!='YES':raise RuntimeError('Prior explicit server consent required')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    with h5py.File(a.dataset) as file:
        meta=json.loads(file['data'].attrs['env_args']);g=file[f'data/{a.demo}']
        initial=g['states'][0];actions=g['actions'][()];truth=g['obs/prop_eef_basepose'][()]
    meta['env_kwargs']['init_curobo']=False
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    env=None;failed=False;audit=None
    report={'mode':'recorded_action_replay_audit','learned_policy':False,'planner_enabled':False,
            'dataset':str(a.dataset),'demo':a.demo,'num_recorded_actions':len(actions),'initial_state_load_count':1,
            'policy_task_success':False,'teacher_action_reads':len(actions),'model_inference_calls':0,
            'requested_reset_physics_steps':a.reset_settle_steps,'reset_physics_steps':0,
            'assisted_constraint_seen':False,'object_pose_write_count':0}
    try:
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs']);env.env.reset()
        if og.sim.dump_state(serialized=True).numel()!=len(initial):raise ValueError('Scene state layout mismatch')
        og.sim.load_state(torch.tensor(initial,dtype=torch.float32),serialized=True)
        audit=ObjectPoseAudit(env.env.task.object_scope)
        report['grasping_mode']=env.robot.grasping_mode
        for _ in range(a.reset_settle_steps):og.sim.step();report['reset_physics_steps']+=1
        before=float(og.sim.current_time)
        for _ in range(3):og.sim.render()
        report['render_only_time_delta_s']=float(og.sim.current_time)-before
        env.env.task._termination_conditions['predicate'].reset(env.env.task,env.env)
        base=[];errors=[];grasp=[False,False];start=time.time()
        with (a.output/'metrics.jsonl').open('w',buffering=1) as log:
            for step,action in enumerate(actions):
                prop=env.robot._get_proprioception_dict();x=as_numpy(env.process_prop_eef_basepose(prop))
                difference=np.abs(x-truth[step]);errors.append(difference)
                position=as_numpy(env.robot.get_position_orientation()[0]).copy();base.append(position)
                record={'step':step,'base_position':position.tolist(),
                     'reference_base_xy':truth[step,:2].tolist(),'reference_base_yaw_rad':float(truth[step,2]),
                     'base_xy_error_m':float(np.linalg.norm(difference[:2])),
                     'base_yaw_error_rad':float(abs((x[2]-truth[step,2]+np.pi)%(2*np.pi)-np.pi)),
                     'max_prop_feature_abs_error_mixed_units':float(difference.max()),'grasped_ever':grasp.copy()}
                if step in (0,1,2,3,5,10,20,50) or step%100==0:
                    log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                env.env.step(torch.tensor(action,dtype=torch.float32))
                report['assisted_constraint_seen']|=any(value is not None for value in env.robot._ag_obj_constraints.values())
                report['completed_actions']=step+1
                prop=env.robot._get_proprioception_dict()
                for index,key in enumerate(('grasp_left','grasp_right')):grasp[index]|=int(as_numpy(prop[key]).reshape(-1)[0])==1
        path=np.stack(base);report.update(goal_success=bool(env.is_success()['task']),grasped_ever=grasp,
             final_goal_status=str(env.env.task._termination_conditions['predicate'].goal_status),
             base_path_length_m=float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()),elapsed_s=time.time()-start)
        np.savez_compressed(a.output/'audit-trace.npz',prop_errors=np.stack(errors),base_positions=path)
    except BaseException as error:
        import traceback
        failed=True;traceback.print_exc();report['error']=repr(error)
    finally:
        if audit:report['object_pose_write_count']=len(audit.events);audit.close()
        (a.output/'audit-result.json').write_text(json.dumps(report,indent=2));print('ACTION_AUDIT_RESULT',json.dumps(report),flush=True)
        try:og.shutdown()
        except SystemExit:pass
    if failed or not report.get('goal_success'):raise SystemExit(7)


if __name__=='__main__':main()
