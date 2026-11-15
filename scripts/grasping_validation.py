"""Explicit grasp-mode checks; no object teleportation or controller changes."""


def require_empty_assisted_grasps(robot):
    if any(value is not None for value in robot._ag_obj_in_hand.values()):
        raise ValueError('Cannot change grasp mode while an assisted object is held')
    if any(value is not None for value in robot._ag_obj_constraints.values()):
        raise ValueError('Existing assisted constraint invalidates physical-only test')


def activate_physical_after_reset(robot):
    """Audit-only override after restoring an empty-handed serialized reset.

    Assisted and physical robot states have different serialized layouts in the
    pinned upstream version. Restore the original layout first, verify no AG
    objects / constraints exist, then disable the assisted branch. Do not use
    this to convert an already-held object or a demonstration mid-state.
    """
    require_empty_assisted_grasps(robot)
    robot._grasping_mode='physical'
    if robot.grasping_mode!='physical':
        raise RuntimeError('Physical mode override did not take effect')
