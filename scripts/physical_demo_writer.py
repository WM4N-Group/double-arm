"""Append only verified physical-policy successes; reset states are stored once."""
import json
from pathlib import Path
import h5py
import numpy as np


def append_physical_success(path, name, initial_state, observations, actions, env_args, report, provenance):
    if provenance.get('grasping_mode')!='physical' or provenance.get('assisted_constraint_seen') is not False:
        raise ValueError('Physical mode with zero assisted constraints required')
    if provenance.get('object_pose_write_count')!=0:
        raise ValueError('Explicit object-pose writes invalidate physical demonstrations')
    if not report.get('sustained_success') or not any(report.get('grasped_ever',[])) or report.get('base_path_length_m',0)<=.5:
        raise ValueError('Successful grasp, transport and sustained goal required')
    actions=np.asarray(actions,dtype=np.float32)
    if actions.ndim!=2 or actions.shape[1]!=21 or len(actions)<30 or not np.isfinite(actions).all():
        raise ValueError('Invalid motor commands')
    required=['prop_state','prop_eef_basepose','base_link_pose','base_qpos','grasp_left','grasp_right']
    if any(key not in observations for key in required):raise ValueError('Missing policy observations')
    for key,value in observations.items():
        if len(value)!=len(actions) or not np.isfinite(value).all():raise ValueError(f'Invalid observations: {key}')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(path,'a') as file:
        data=file.require_group('data')
        if name in data:raise FileExistsError(name)
        existing=data.attrs.get('env_args')
        if existing and json.loads(existing)['env_name']!=env_args['env_name']:raise ValueError('Cannot mix tasks')
        data.attrs['env_args']=json.dumps(env_args);data.attrs['demonstration_grasping_mode']='physical'
        data.attrs['schema']='physical_policy_successes_with_single_reset_state_v1'
        data.attrs['collection_method']='closed-loop learned memory policy in physical mode; no online planner or teacher fallback'
        pending='_pending_'+name
        if pending in data:raise FileExistsError(pending)
        group=data.create_group(pending);group.attrs['partial']=True
        group.create_dataset('initial_state',data=np.asarray(initial_state,dtype=np.float32),compression='gzip')
        group.create_dataset('actions',data=actions,compression='gzip')
        obs=group.create_group('obs')
        for key,value in observations.items():obs.create_dataset(key,data=np.asarray(value,dtype=np.float32),compression='gzip')
        group.attrs['provenance']=json.dumps(provenance);group.attrs['evaluation']=json.dumps(report)
        group.attrs['reset_state_layout']='configured grasp mode, then explicit physical override while empty-handed'
        group.attrs['partial']=False;file.flush();data.move(pending,name)
        data.attrs['total']=sum(len(data[key]['actions']) for key in data if key.startswith('demo_'))
        file.flush()
