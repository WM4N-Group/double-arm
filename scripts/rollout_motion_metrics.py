"""Supplementary evaluation-only motion metrics, never policy observations."""
import numpy as np


def cup_motion_metrics(cup_positions,base_positions,cup_grasped,initial_cup_position,min_lift=.03):
    """Conservative transport evidence from consecutive post-step grasp reports.

    This does not decide task acceptance. Large cup steps alone do not prove a
    teleport: physical impacts can also cause rapid movement; inspect the video
    together with the explicit pose-write / assisted-constraint audits.
    """
    cups=np.asarray(cup_positions,dtype=np.float64)
    bases=np.asarray(base_positions,dtype=np.float64)
    grasped=np.asarray(cup_grasped,dtype=bool)
    initial=np.asarray(initial_cup_position,dtype=np.float64)
    if cups.ndim!=2 or cups.shape[1]!=3 or len(cups)<1:raise ValueError('Expected post-step cup XYZ trace')
    if bases.shape!=cups.shape or grasped.shape!=(len(cups),) or initial.shape!=(3,):raise ValueError('Misaligned motion audit')
    if not all(np.isfinite(value).all() for value in (cups,bases,initial)):raise ValueError('Nonfinite motion audit')
    cup_steps=np.linalg.norm(np.diff(np.vstack([initial,cups]),axis=0),axis=1)
    base_steps=np.linalg.norm(np.diff(bases[:,:2],axis=0),axis=1)
    supported=grasped & (cups[:,2]-initial[2]>min_lift)
    consecutive=supported[1:] & supported[:-1]
    return {'grasp_supported_base_path_m':float(base_steps[consecutive].sum()),
            'cup_path_length_m':float(cup_steps.sum()),
            'max_cup_step_displacement_m':float(cup_steps.max()),
            'cup_grasp_report_and_lift_frames':int(supported.sum()),
            'transport_metric_scope':'Consecutive post-step cup-specific grasp report and >3cm lift; consult physical/assisted mode; supplementary, not a new acceptance gate'}
