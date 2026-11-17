import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi0_single_demo_fit import transition_indices,frame_schedule,sequence_sha256,storage_cast,variant_freeze_filter,inference_payload,verify_postfit_repair


class FitPlanTests(unittest.TestCase):
    def test_frames_are_real_unmodified_source_horizons(self):
        self.assertEqual(transition_indices(1714,636),[606,621,631,635,636,637,641,651,666])
        for args in ((100,10),(100,60),(100,True)):
            with self.assertRaises(ValueError):transition_indices(*args)

    def test_matching_balanced_deterministic_schedule_and_bound(self):
        indices=transition_indices(1714,636);a=frame_schedule(indices,2000);b=frame_schedule(indices,2000)
        self.assertEqual(a,b);self.assertEqual(sequence_sha256(a),sequence_sha256(b))
        for i in range(0,1998,9):self.assertEqual(set(a[i:i+9]),set(indices))
        with self.assertRaises(ValueError):frame_schedule(indices,2001)
        with self.assertRaises(ValueError):frame_schedule([1]*9,20)

    def test_common_storage_cast_is_independent_from_gradient_filter(self):
        try:
            import jax.numpy as jnp
            from flax import nnx
            from openpi.shared.nnx_utils import PathRegex
        except ImportError:
            self.skipTest('Real NNX fixture runs in server openpi env')
        class Leaf(nnx.Module):
            def __init__(self):self.kernel=nnx.Param(jnp.array([1.003],jnp.float32))
        class Encoder(nnx.Module):
            def __init__(self):self.img=Leaf();self.llm=Leaf()
        class Tiny(nnx.Module):
            def __init__(self):self.PaliGemma=Encoder()
        params=nnx.state(Tiny());common=PathRegex(r'.*PaliGemma/llm/.*')
        a=storage_cast(params,common);b=storage_cast(params,common)
        self.assertEqual(str(a.PaliGemma.img.kernel.value.dtype),'float32')
        self.assertEqual(str(b.PaliGemma.img.kernel.value.dtype),'float32')
        self.assertEqual(str(a.PaliGemma.llm.kernel.value.dtype),'bfloat16')
        self.assertEqual(len(a.filter(variant_freeze_filter(common,'original_vision_trainable')).flat_state()),1)
        self.assertEqual(len(b.filter(variant_freeze_filter(common,'extra_vision_frozen')).flat_state()),2)
        with self.assertRaises(ValueError):variant_freeze_filter(common,'implicit')

    def test_real_tiny_init_matches_params_and_dtype_across_freeze_filters(self):
        try:
            import jax
            import jax.numpy as jnp
            from flax import nnx
            from openpi.shared.nnx_utils import PathRegex
            from openpi.training import optimizer,sharding
        except ImportError:
            self.skipTest('Real NNX init fixture runs on server CPU')
        from types import SimpleNamespace
        from pi0_single_demo_fit import initialize_state
        from preflight_pi0_accumulation_gpu import trainable_fingerprint
        class Leaf(nnx.Module):
            def __init__(self,key):self.kernel=nnx.Param(jax.random.normal(key,(2,),dtype=jnp.float32))
        class Encoder(nnx.Module):
            def __init__(self,key):self.img=Leaf(key);self.llm=Leaf(jax.random.fold_in(key,1))
        class Tiny(nnx.Module):
            def __init__(self,key):self.PaliGemma=Encoder(key)
        common=PathRegex(r'.*PaliGemma/llm/.*');states=[]
        for variant in ('original_vision_trainable','extra_vision_frozen'):
            freeze=variant_freeze_filter(common,variant)
            cfg=SimpleNamespace(model=SimpleNamespace(create=lambda key:Tiny(key)),
                optimizer=optimizer.AdamW(),lr_schedule=optimizer.CosineDecaySchedule(),
                trainable_filter=nnx.All(nnx.Param,nnx.Not(freeze)),weight_loader=None)
            fake_upstream=SimpleNamespace(_load_weights_and_validate=lambda loader,shape:{})
            state,_=initialize_state(cfg,common,jax.random.key(42),sharding.make_mesh(1),fake_upstream)
            states.append(state)
        self.assertEqual(trainable_fingerprint(states[0].params),trainable_fingerprint(states[1].params))
        self.assertEqual(str(states[1].params.PaliGemma.img.kernel.value.dtype),'float32')
        self.assertEqual(str(states[1].params.PaliGemma.llm.kernel.value.dtype),'bfloat16')
        before=trainable_fingerprint(states[0].params)
        params,graph,step=inference_payload(states[0])
        states.clear();del state
        import gc
        gc.collect()
        self.assertEqual(step,0)
        self.assertEqual(trainable_fingerprint(params),before)
        def infer(p):
            model=nnx.merge(graph,p);model.eval()
            return model.PaliGemma.img.kernel.value*2
        prediction=jax.jit(infer)(params)
        self.assertEqual(prediction.shape,(2,))
        self.assertTrue(bool(jnp.isfinite(prediction).all()))

    def test_postfit_repeat_requires_explicit_verified_completed_fit_evidence(self):
        import copy
        fixture={'stage':'single_demo_fit_failed','actual_optimizer_updates':2000,
            'optimizer_and_lr_counts':{'adam':2000,'lr':2000},'samples':[{'stage':'initial'}]*27,
            'checkpoint_saved':False,'variant':'original_vision_trainable',
            'error':"TypeCheckError opt_state Actual value: None",
            'training_log':[{'update':2000,'loss':.001,'grad_norm':.2,'param_norm':1300}],
            'initial_full_parameter_sha256':'a'*64,'final_trainable_sha256':'b'*64,
            'initial_trainable_sha256':'c'*64}
        self.assertEqual(verify_postfit_repair(fixture),'a'*64)
        for key,value in [('actual_optimizer_updates',1999),('error','OOM'),('optimizer_and_lr_counts',{}),
                          ('variant','extra_vision_frozen'),('samples',[{'stage':'final'}]*27)]:
            changed=copy.deepcopy(fixture);changed[key]=value
            with self.assertRaises(ValueError):verify_postfit_repair(changed)


if __name__=='__main__':unittest.main()
