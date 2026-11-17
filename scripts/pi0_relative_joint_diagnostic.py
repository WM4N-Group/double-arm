"""NEW diagnostic encoding: native base/grippers, observation-relative joints.

Original21D proprio and real3 RGB are unchanged. Subtract the CURRENT input
joint state from each future target; inverse adds that SAME chunk-anchor state,
not future states, odometry, teacher, phase, or a per-step moving anchor.
Never use these action stats/outputs with old native checkpoints or serving.
"""
import hashlib
JOINT_ACTION_INDICES=tuple(range(3,13))+tuple(range(14,20))
JOINT_STATE_INDICES=tuple(range(3,13))+tuple(range(13,19))
PROFILE='relative_joints_meanstd_v1'


def joint_encoding(actions,state,inverse=False):
    import numpy as np
    actions=np.asarray(actions,dtype=np.float32);state=np.asarray(state,dtype=np.float32)
    if actions.ndim!=2 or actions.shape[1]!=21 or state.shape!=(21,) or not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError('Finite original21D state and horizon x21 actions required')
    result=actions.copy();a=list(JOINT_ACTION_INDICES);q=list(JOINT_STATE_INDICES)
    if inverse:result[:,a]+=state[q]
    else:result[:,a]-=state[q]
    return result


class RelativeJointInputs:
    def __call__(self,data):
        from momagen_pi0_schema import make_pi0_inputs
        result=make_pi0_inputs(data)
        if 'actions' in result:result['actions']=joint_encoding(result['actions'],result['state'])
        return result


class RelativeJointOutputs:
    def __call__(self,data):
        import numpy as np
        from momagen_pi0_schema import unpad_pi0_actions
        # Policy output contains the same observed, normalized-then-unnormalized
        # state supplied at chunk start; NEVER look up source labels here.
        state=np.asarray(data['state'],dtype=np.float32)
        if state.ndim!=1 or len(state)<21:raise ValueError('Chunk-anchor observation state required')
        return {'actions':joint_encoding(unpad_pi0_actions(data['actions']),state[:21],inverse=True)}


def relative_joint_stats(provenance,original_actions_stats):
    """All16 TRAIN, actual50-step label windows; retain base/gripper stats EXACT.

    No images/model/simulator or dataset writes. Action quantiles are explicitly
    unknown/unused; this profile is mean/std, NOT the paper quantile recipe.
    """
    import h5py
    import numpy as np
    if provenance.get('split')!='train' or len(provenance.get('episodes',[]))!=16:
        raise ValueError('Original16 TRAIN provenance only, never heldout')
    total=0;sum_x=np.zeros(16,np.float64);sum_x2=np.zeros(16,np.float64);sources=[]
    aa=list(JOINT_ACTION_INDICES);ss=list(JOINT_STATE_INDICES)
    for entry in provenance['episodes']:
        with h5py.File(entry['source']) as file:
            g=file['data/'+entry['source_demo']];actions=g['actions'][:];states=g['obs/prop_state'][:]
            fingerprint=hashlib.sha256(actions.tobytes()+states[0].tobytes()).hexdigest()
            if fingerprint!=entry['fingerprint'] or len(actions)!=entry['frames']:
                raise ValueError('Immutable training source changed')
            for offset in range(50):
                indices=np.minimum(np.arange(len(actions))+offset,len(actions)-1)
                # Match actual float32 label subtraction BEFORE normalization.
                residual=(actions[indices][:,aa].astype(np.float32)-states[:,ss].astype(np.float32)).astype(np.float64)
                sum_x+=residual.sum(axis=0);sum_x2+=(residual**2).sum(axis=0);total+=len(actions)
            sources.append({'source':entry['source'],'source_demo':entry['source_demo'],'fingerprint':fingerprint,'frames':len(actions)})
    mean=sum_x/total;std=np.sqrt(np.maximum(sum_x2/total-mean**2,0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():raise ValueError('Nonfinite stats')
    new_mean=np.asarray(original_actions_stats['mean'],np.float64).copy()
    new_std=np.asarray(original_actions_stats['std'],np.float64).copy()
    if new_mean.shape!=(21,) or new_std.shape!=(21,):raise ValueError('Expected native action stats')
    new_mean[aa]=mean;new_std[aa]=std
    return {'mean':new_mean.tolist(),'std':new_std.tolist(),'q01':None,'q99':None}, {
        'source_split':'train','source_episodes':sources,'label_values_per_dimension':total,
        'scope':'All16 original TRAIN x50 episode-clamped future labels, each relative to CURRENT state; no split/dataset writes',
        'retained_original_norm_axes':[0,1,2,13,20],'changed_norm_axes':aa,
        'no_quantile_claim':True,'action_encoding_and_joint_norm_are_one_explicit_recipe_variant':True}
