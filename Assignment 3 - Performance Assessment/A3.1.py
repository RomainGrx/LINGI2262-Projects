#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 Mar 16, 15:58:39
@last modified : 2021 Mar 16, 16:54:22
"""

import random
import numpy as np
from functools import partial
from multiprocessing import Pool


def after_how_much(probability, number_failed):
    cnt = 0
    iterator = 0
    while cnt < number_failed:
        iterator += 1
        if random.random() > p:
            cnt += 1
    return iterator


def run_n_samples(n_samples, probability, number_failed):
    with Pool() as pool:
        samples = pool.map(
            partial(after_how_much, number_failed=number_failed),
            [probability] * n_samples,
        )
    return samples


n_samples = 100000000
p = 0.8

samples = run_n_samples(n_samples, p, 1)
mu = np.mean(samples)
std = np.std(samples)

print(f"samples stats until first error :: \n\tmean -> {mu:.3f} | std -> {std:.3f}") # 5, 4.472

samples = run_n_samples(n_samples, p, 3)
mu = np.mean(samples)
std = np.std(samples)

print(f"samples stats until third errors :: \n\tmean -> {mu:.3f} | std -> {std:.3f}") # 15, 
