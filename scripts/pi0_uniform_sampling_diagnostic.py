"""Explicit sampler-only B condition; encoding/norm/model remain frozen A profile."""
import random

VARIANT='uniform_full_trajectory_same_base_v1'
RECIPE='100% shuffled complete-demo cycles; no transition oversampling; not paper recipe'


def uniform_schedule(length,onset,updates=20000,seed=42):
    # onset retained only for signature compatibility; never used by sampler or
    # model. No phase/time/demo index becomes a policy input.
    if type(length) is not int or type(updates) is not int or not 1<=length<=updates<=100000:
        raise ValueError('Bounded complete-trajectory sampling required')
    rng=random.Random(seed);order=[]
    while len(order)<updates:
        pool=list(range(length));rng.shuffle(pool)
        while pool and len(order)<updates:order.append(pool.pop())
    return order
