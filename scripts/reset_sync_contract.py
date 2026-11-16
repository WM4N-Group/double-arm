"""Fail-closed reset RGB integration gate, NOT a task-success criterion."""
import math

CAMERAS=('head_image','left_wrist_image','right_wrist_image')
LIMITS={'rgb_mae_0_255':15.0,'root_translation_m':.02,'root_yaw_rad':.05,
        'joint_max_abs_rad':.05,'minimum_xy_rotation_determinant':.98}


def select_reset_sync_candidate(report):
    if (report.get('mode')!='reset_sync_image_audit_only' or report.get('grasping_mode')!='physical'
        or report.get('initial_state_load_count')!=1 or report.get('teacher_action_reads')!=0
        or report.get('simulator_control_calls')!=0 or report.get('inference_calls')!=0
        or report.get('assisted_constraint_seen') is not False or report.get('object_pose_write_count')!=0
        or report.get('series_completed') is not True):return None
    eligible=[]
    for item in report.get('snapshots',[]):
        step=item.get('physics_steps')
        if type(step) is not int or not 0<=step<=20:continue
        rgb=item.get('rgb_mae_0_255',{})
        if set(rgb)!=set(CAMERAS):continue
        values=list(rgb.values())+[item.get(key) for key in ('root_translation_m','root_yaw_abs_difference_rad','joint_max_abs_difference_rad','xy_rotation_determinant')]
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in values):continue
        if (all(0<=value<=LIMITS['rgb_mae_0_255'] for value in rgb.values())
            and 0<=item['root_translation_m']<=LIMITS['root_translation_m']
            and 0<=item['root_yaw_abs_difference_rad']<=LIMITS['root_yaw_rad']
            and 0<=item['joint_max_abs_difference_rad']<=LIMITS['joint_max_abs_rad']
            and LIMITS['minimum_xy_rotation_determinant']<=item['xy_rotation_determinant']<=1.000001):eligible.append(step)
    return min(eligible) if eligible else None
