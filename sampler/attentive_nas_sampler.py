# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import atexit
import os
import random
import copy

def count_helper(v, flops, m):
    if flops not in m:
        m[flops] = {}
    if v not in m[flops]:
        m[flops][v] = 0
    m[flops][v] += 1 


def round_flops(flops, step):
    return int(round(flops / step) * step)


def convert_count_to_prob(m):
    if isinstance(m[list(m.keys())[0]], dict):
        for k in m:
            convert_count_to_prob(m[k])
    else:
        t = sum(m.values())
        for k in m:
            m[k] = 1.0 * m[k] / t


def sample_helper(flops, m):
    keys = list(m[flops].keys())
    probs = list(m[flops].values())
    return random.choices(keys, weights=probs)[0]


def build_trasition_prob_matrix(file_handler, step):
    # initlizie
    prob_map = {}
    prob_map['discretize_step'] = step
    for k in ['flops', 'resolution', 'width', 'depth', 'kernel_size', 'expand_ratio']:
        prob_map[k] = {}

    cc = 0
    for line in file_handler:
        vals = eval(line.strip())

        # discretize
        flops = round_flops(vals['flops'], step)
        prob_map['flops'][flops] = prob_map['flops'].get(flops, 0) + 1

        # resolution
        r = vals['resolution']
        count_helper(r, flops, prob_map['resolution'])

        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            for idx, v in enumerate(vals[k]):
                if idx not in prob_map[k]:
                    prob_map[k][idx] = {}
                count_helper(v, flops, prob_map[k][idx])

        cc += 1

    # convert count to probability
    for k in ['flops', 'resolution', 'width', 'depth', 'kernel_size', 'expand_ratio']:
        convert_count_to_prob(prob_map[k])
    prob_map['n_observations'] = cc
    return prob_map



class ArchSampler():
    def __init__(self, arch_to_flops_map_file_path, discretize_step, model, acc_predictor=None):
        super(ArchSampler, self).__init__()

        with open(arch_to_flops_map_file_path, 'r') as fp:
            self.prob_map = build_trasition_prob_matrix(fp, discretize_step)

        self.discretize_step = discretize_step
        self.model = model

        self.acc_predictor = acc_predictor

        self.min_flops = min(list(self.prob_map['flops'].keys()))
        self.max_flops = max(list(self.prob_map['flops'].keys()))

        self.curr_sample_pool = None #TODO; architecture samples could be generated in an asynchronous way


    def sample_one_target_flops(self, flops_uniform=False):
        f_vals = list(self.prob_map['flops'].keys())
        f_probs = list(self.prob_map['flops'].values())

        if flops_uniform:
            return random.choice(f_vals)
        else:
            return random.choices(f_vals, weights=f_probs)[0]


    def sample_archs_according_to_flops(self, target_flops,  n_samples=1, max_trials=100, return_flops=True, return_trials=False):
        archs = []
        #for _ in range(n_samples):
        while len(archs) < n_samples:
            for _trial in range(max_trials+1):
                arch = {}
                arch['resolution'] = sample_helper(target_flops, self.prob_map['resolution'])
                for k in ['width', 'kernel_size', 'depth', 'expand_ratio']:
                    arch[k] = []
                    for idx in sorted(list(self.prob_map[k].keys())):
                        arch[k].append(sample_helper(target_flops, self.prob_map[k][idx]))
                if self.model:
                    self.model.set_active_subnet(**arch)
                    flops = self.model.compute_active_subnet_flops()
                    if return_flops:
                        arch['flops'] = flops
                    if round_flops(flops, self.discretize_step) == target_flops:
                        break
                else:
                    raise NotImplementedError
            #accepte the sample anyway
            archs.append(arch)
        return archs    


