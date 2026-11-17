from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi0_relative_full_profile import full_schedule,validate_fields,PROFILE,ASSET_ID,MODEL_ENCODING,RUNTIME_FILES


class FullProfileTests(unittest.TestCase):
    def test_full_coverage_and_explicit_transition_mix(self):
        order=full_schedule(1714,636);self.assertEqual(len(order),20000);self.assertEqual(set(order),set(range(1714)))
        self.assertEqual(order,full_schedule(1714,636))
        self.assertEqual(set(order[1::2]),{606,621,631,635,636,637,641,651,666})
        with self.assertRaises(ValueError):full_schedule(1714,636,2000)

    def contract(self):
        return {'schema_version':1,'policy_profile':PROFILE,'model':'pi0','lora_rank':32,
            'model_action_encoding':MODEL_ENCODING,'motor_action_encoding':'native_21_after_inverse',
            'normalization_mode':'mean_std','norm_stats_asset_id':ASSET_ID,'original_proprio_dim':21,
            'model_action_dim':32,'action_horizon':50,'parameter_restore_dtype':'preserve_saved',
            'training_is_diagnostic':True,'paper_reproduction_complete':False,'training_completed':True,
            'train_demonstrations':1,'optimizer_updates':20000,'training_source_demo':'demo_0','train_frames':1714,
            'optimizer_and_lr_counts':{'adam':20000,'lr':20000},'norm_stats_sha256':'a'*64,
            'full_parameter_sha256':'b'*64,'frame_schedule_sha256':'c'*64,
            'runtime_source_sha256':{name:'d'*64 for name in RUNTIME_FILES},
            'openpi_revision':'215abfb217dbac7d5f1273282331b9b1866c0479',
            'teacher_enabled':False,'planner_enabled':False,'object_pose_inputs':False,
            'parameter_dtype_counts':{'float32':495908624,'bfloat16':2819996672}}

    def test_legacy_or_partial_or_wrong_norm_never_accepted(self):
        c=self.contract();self.assertEqual(validate_fields(c),c)
        for key,value in [('policy_profile','native'),('normalization_mode','quantile'),('optimizer_updates',2000),
                          ('training_completed',False),('parameter_restore_dtype','bfloat16'),('paper_reproduction_complete',True)]:
            x=self.contract();x[key]=value
            with self.assertRaises(ValueError):validate_fields(x)

    def test_real_tiny_official_checkpoint_preserves_params_and_norms(self):
        try:
            import jax
            import jax.numpy as jnp
            import numpy as np
            import optax
            from flax import nnx
            from openpi.training import checkpoints,utils
            from openpi.models.model import restore_params
            from openpi.shared.normalize import NormStats
        except ImportError:
            self.skipTest('Real Orbax/NNX tiny fixture runs on server CPU')
        import dataclasses
        import json
        import tempfile
        from types import SimpleNamespace
        class Toy(nnx.Module):
            def __init__(self):
                self.linear=nnx.Linear(2,1,rngs=nnx.Rngs(42))
                self.low_precision=nnx.Param(jnp.asarray([.125],jnp.bfloat16))
        toy=Toy();params=nnx.state(toy);tx=optax.adamw(1e-3)
        state=utils.TrainState(step=0,params=params,model_def=nnx.graphdef(toy),
            opt_state=tx.init(params),tx=tx,ema_decay=None,ema_params=None)
        grad=jax.tree.map(jnp.ones_like,state.params)
        updates,opt=tx.update(grad,state.opt_state,state.params)
        nnx.update(toy,optax.apply_updates(state.params,updates))
        state=dataclasses.replace(state,step=1,params=nnx.state(toy),opt_state=opt)
        norms={'state':NormStats(np.zeros(2),np.ones(2)),'actions':NormStats(np.zeros(1),np.ones(1))}
        data=SimpleNamespace(asset_id='fixture/tiny',norm_stats=norms)
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)/'new-checkpoints'
            manager,resume=checkpoints.initialize_checkpoint_dir(destination,keep_period=1,overwrite=False,resume=False)
            self.assertFalse(resume)
            checkpoints.save_state(manager,state,SimpleNamespace(data_config=lambda:data),step=0)
            manager.wait_until_finished();manager.close()
            cp=destination/'0'
            self.assertTrue(json.loads((cp/'_CHECKPOINT_METADATA').read_text())['commit_timestamp_nsecs'])
            restored=restore_params(cp/'params',restore_type=np.ndarray,dtype=None)
            jax.tree.map(np.testing.assert_array_equal,state.params.to_pure_dict(),restored)
            self.assertEqual(str(restored['low_precision'].dtype),'bfloat16')
            saved=checkpoints.load_norm_stats(cp/'assets','fixture/tiny')
            for key in norms:np.testing.assert_array_equal(saved[key].std,norms[key].std)


if __name__=='__main__':unittest.main()
