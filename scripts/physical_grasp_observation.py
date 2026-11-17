"""Read-only NO-ASSISTANCE evidence for FUTURE physical diagnostics.

No simulator import/setter, controller change, VLA input, or task-success gate.
A valid mode/empty Python dictionaries alone are insufficient: a read-only USD
ag_constraint presence query is also required. Unknown is -1, never false/0.
"""
import hashlib
from pathlib import Path

ARMS=('left','right')
FIELDS=('constraint_active','object_in_hand','gripper_freeze','usd_ag_constraint_present')
PINNED_MANIPULATION_SHA256='af1e0245078a69a67368be4e70af4f5b79190c7d7dcff4fe9be210da1c39b0ba'


def snapshot_no_assistance(robot,prim_exists=None,inactive_freeze_none_source=None):
    result={'source':'READ_ONLY_NO_ASSISTANCE_EVIDENCE_NOT_POLICY_INPUT_OR_GRASP_SUCCESS',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'grasping_mode':None,'errors':[],'gripper_freeze_raw':{arm:'unobserved' for arm in ARMS},
            **{name:{arm:-1 for arm in ARMS} for name in FIELDS}}
    # None is NOT generally equivalent to false. This one installed source
    # initializes freeze=None, then deploy_control uses `if freeze[arm]`.
    # Permit that KNOWN inactive sentinel only after exact read-only source SHA
    # validation; preserve raw None and provenance, never change robot state.
    none_inactive=False
    if inactive_freeze_none_source is not None:
        try:
            digest=hashlib.sha256(Path(inactive_freeze_none_source).read_bytes()).hexdigest()
            if digest!=PINNED_MANIPULATION_SHA256:raise ValueError('Unaudited manipulation source SHA')
            none_inactive=True
            result['freeze_none_inactive_source_sha256']=digest
            result['freeze_none_scope']='Exact pinned __init__ None sentinel; deploy_control truth-test is inactive. Raw None retained; no field mutation.'
        except Exception as error:result['errors'].append('freeze_none_source: '+type(error).__name__+': '+str(error))
    try:
        mode=robot.grasping_mode
        if not isinstance(mode,str):raise TypeError('Unknown/non-string grasping mode')
        result['grasping_mode']=mode
    except Exception as error:result['errors'].append('grasping_mode: '+type(error).__name__+': '+str(error))
    for attribute,field in (('_ag_obj_constraints','constraint_active'),('_ag_obj_in_hand','object_in_hand'),('_ag_freeze_gripper','gripper_freeze')):
        try:
            values=getattr(robot,attribute)
            if not isinstance(values,dict):raise TypeError('Expected actual arm mapping')
        except Exception as error:
            result['errors'].append(attribute+': '+type(error).__name__+': '+str(error));continue
        for arm in ARMS:
            try:
                value=values[arm]
                if field=='gripper_freeze':
                    result['gripper_freeze_raw'][arm]=value if value is None or type(value) is bool else 'unsupported_type:'+type(value).__name__
                    if value is None and none_inactive:
                        result[field][arm]=0
                    elif type(value) is bool:
                        result[field][arm]=int(value)
                    else:raise TypeError('Freeze unknown; require Boolean or source-verified inactive None sentinel')
                else:
                    # Never invoke __bool__ on an engine object/prim.
                    result[field][arm]=int(value is not None)
            except Exception as error:result['errors'].append(attribute+'.'+arm+': '+type(error).__name__+': '+str(error))
    for arm in ARMS:
        try:
            if prim_exists is None:raise ValueError('No actual read-only USD presence query supplied')
            path=robot.eef_links[arm].prim_path+'/ag_constraint'
            present=prim_exists(path)
            if type(present) is not bool:raise TypeError('USD query must return known Boolean')
            result['usd_ag_constraint_present'][arm]=int(present)
        except Exception as error:result['errors'].append('USD.'+arm+': '+type(error).__name__+': '+str(error))
    return result


def no_assistance_confirmed(report):
    """Only absence-of-assistance evidence, NOT stable physical grasp success."""
    if (report.get('grasping_mode')!='physical' or not isinstance(report.get('errors'),list) or report['errors']):
        return False
    return all(isinstance(report.get(field),dict) and set(report[field])==set(ARMS)
        and all(type(report[field][arm]) is int and report[field][arm]==0 for arm in ARMS) for field in FIELDS)
