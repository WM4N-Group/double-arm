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
