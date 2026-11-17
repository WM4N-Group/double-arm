"""Read-only NO-ASSISTANCE evidence for FUTURE physical diagnostics.

No simulator import/setter, controller change, VLA input, or task-success gate.
A valid mode/empty Python dictionaries alone are insufficient: a read-only USD
ag_constraint presence query is also required. Unknown is -1, never false/0.
"""
import hashlib
from pathlib import Path

ARMS=('left','right')
FIELDS=('constraint_active','object_in_hand','gripper_freeze','usd_ag_constraint_present')


def snapshot_no_assistance(robot,prim_exists=None):
    result={'source':'READ_ONLY_NO_ASSISTANCE_EVIDENCE_NOT_POLICY_INPUT_OR_GRASP_SUCCESS',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'grasping_mode':None,'errors':[],**{name:{arm:-1 for arm in ARMS} for name in FIELDS}}
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
                    if type(value) is not bool:raise TypeError('Expected recorded Boolean, not an unknown truthy object')
                    result[field][arm]=int(value)
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
