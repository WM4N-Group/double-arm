"""Paper C.1 endpoint predicate, offline audit only; no live dataset mutation.

Compare q[i] to q[i+lag], not all intermediate adjacent differences. Inclusive
window deletion is an explicit indexing convention. Which author joint subset
was used remains unverified; keep source-field provenance in the report.
"""
import numpy as np


def frozen_segment_mask(joint_positions,lag=5,epsilon=1e-3):
    q=np.asarray(joint_positions,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]<1 or not np.isfinite(q).all():raise ValueError('Finite joint-position matrix required')
    if type(lag) is not int or lag<1 or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('Invalid endpoint predicate')
    endpoint_quiet=np.all(np.abs(q[lag:]-q[:-lag])<epsilon,axis=1) if len(q)>lag else np.zeros(0,bool)
    starts=np.flatnonzero(endpoint_quiet)
    difference=np.zeros(len(q)+1,np.int64)
    np.add.at(difference,starts,1);np.add.at(difference,starts+lag+1,-1)
    return np.cumsum(difference[:-1])>0,endpoint_quiet
