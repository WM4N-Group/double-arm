"""Isolated batch-count arithmetic / optimizer wrapper; NOT wired into training."""


def accumulation_plan(microbatch,effective_batch,optimizer_updates,trainable_parameters=495908624):
    values=(microbatch,effective_batch,optimizer_updates,trainable_parameters)
    if any(type(n) is not int or n<=0 for n in values) or effective_batch%microbatch:
        raise ValueError('Positive integral batch sizes with exact accumulation required')
    k=effective_batch//microbatch
    return {'microbatch':microbatch,'effective_batch':effective_batch,'accumulation_steps':k,
        'optimizer_updates':optimizer_updates,'required_micro_steps':optimizer_updates*k,
        'schedule_time_unit':'optimizer_updates_not_micro_steps','trainable_parameters_assumed':trainable_parameters,
        'sample_exposures':optimizer_updates*effective_batch,
        'extra_fp32_gradient_accumulator_bytes_lower_bound':trainable_parameters*4,
        'extra_fp32_gradient_accumulator_gib_lower_bound':trainable_parameters*4/2**30,
        'pi0_gpu_memory_fit_verified':False,'wired_into_active_training':False}


def accumulating_optimizer(inner,steps):
    if type(steps) is not int or steps<1:raise ValueError('Positive integral accumulation steps required')
    import optax
    # The inner transformation includes clipping + AdamW; clip AFTER averaging
    # gradients, not each microbatch independently. Its schedule counts updates.
    return optax.MultiSteps(inner,every_k_schedule=steps,use_grad_mean=True)


class HostGradientAccumulator:
    """FP32 raw gradient sums in host RAM. No clipping/optimizer/micro updates.

    Lazy numerical import keeps workflow gates stdlib-only. All leaves are
    validated BEFORE any accumulator mutation, so failed adds are transactional.
    """
    def __init__(self, steps):
        if type(steps) is not int or steps < 1:
            raise ValueError('Positive integral accumulation steps required')
        self.steps = steps
        self.count = 0
        self._sums = None
        self._averaged = False

    @property
    def nbytes(self):
        return 0 if self._sums is None else sum(leaf.nbytes for leaf in self._sums)

    def add(self, leaves):
        import numpy as np
        if self._averaged or self.count >= self.steps:
            raise ValueError('Accumulation window already complete')
        values = [np.asarray(leaf) for leaf in leaves]
        if not values or any(leaf.dtype != np.float32 or not np.isfinite(leaf).all() for leaf in values):
            raise ValueError('Finite FP32 raw host gradient leaves required')
        if self._sums is not None and (len(values) != len(self._sums)
            or any(a.shape != b.shape for a, b in zip(values, self._sums))):
            raise ValueError('Gradient structure/shape changed inside accumulation window')
        if self._sums is None:
            self._sums = [np.zeros_like(leaf) for leaf in values]
        for total, leaf in zip(self._sums, values):
            np.add(total, leaf, out=total)
        self.count += 1

    def mean(self):
        import numpy as np
        if self.count != self.steps:
            raise ValueError('No optimizer update before full accumulation window')
        if not self._averaged:
            for total in self._sums:
                np.divide(total, np.float32(self.steps), out=total)
            self._averaged = True
        return self._sums
