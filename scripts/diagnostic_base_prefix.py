"""OFFLINE replay-only intervention; never a live policy adapter or model input."""
PREFIX_STEPS=682
REPLAY_STEPS=1200


def replace_initial_base_only(recorded,source):
    import numpy as np
    recorded=np.asarray(recorded);source=np.asarray(source)
    if recorded.shape!=(3000,21) or source.shape!=(1714,21):raise ValueError('Actual failed B tape and exact TRAIN source required')
    if recorded.dtype!=np.float32 or source.dtype!=np.float32 or not np.isfinite(recorded).all() or not np.isfinite(source).all():raise ValueError('Unmodified finite float32 native commands required')
    result=recorded[:REPLAY_STEPS].copy();result[:PREFIX_STEPS,:3]=source[:PREFIX_STEPS,:3]
    np.testing.assert_array_equal(result[:,3:],recorded[:REPLAY_STEPS,3:])
    np.testing.assert_array_equal(result[PREFIX_STEPS:,:3],recorded[PREFIX_STEPS:REPLAY_STEPS,:3])
    return result
