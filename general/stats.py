import pathlib
import json
import sys
import tensorflow as tf
import wandb

class Stats(dict):
    def __init__(self, init_dict=None, *args, **kwargs):
        if init_dict is not None:
            self._dict = {}
            for k, v in init_dict.items():
                self._dict[k] = v
        else:
            self._dict = dict(**kwargs)
        self._stats_to_track_dict = {}

    def record_stats(self, names, prefix='', **kwargs):
        log = {}
        for name in names:
            log[prefix + name] = self._dict[name]

        for k, v in kwargs.items():
            log[k] = v

        wandb.log(log)
   
    def register_var(self, name, stats_to_track):
        self._stats_to_track_dict[name] = []

        if 'std' in stats_to_track:
            stats_to_track.append('sq_sum')
            stats_to_track.append('num')
            stats_to_track.append('avg')
        elif 'avg' in stats_to_track:
            stats_to_track.append('num')

        for stat in stats_to_track:
            if stat == 'max' :
                self._dict[f'{stat}_{name}'] = float('-inf')
            elif stat == 'min' :
                self._dict[f'{stat}_{name}'] = float('inf')
            self._dict[f'{stat}_{name}'] = 0
            self._stats_to_track_dict[name].append(stat)

    def record_var(self, name, val):
        stat_list = self._stats_to_track_dict[name]
        
        if 'max' in stat_list:
            self._dict[f'max_{name}'] = max(val, self._dict[f'max_{name}'])
        if 'min' in stat_list:
            self._dict[f'min_{name}'] = min(val, self._dict[f'min_{name}'])
        if 'avg' in stat_list:
            self._dict[f'avg_{name}'] = (self._dict[f'num_{name}'] * self._dict[f'avg_{name}'] + val) / (self._dict[f'num_{name}'] + 1)
        if 'sum' in stat_list:
            self._dict[f'sum_{name}'] += val
        if 'sq_sum' in stat_list:
            self._dict[f'sq_sum_{name}'] += val ** 2
        if 'num' in stat_list:
            self._dict[f'num_{name}'] += 1
        if 'std' in stat_list:
            self._dict[f'std_{name}'] = np.sqrt(max(self._dict[f'sq_sum_{name}'] / self._dict[f'num_{name}'] - self._dict[f'avg_{name}'] ** 2, 0))

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getitem__(self, key):
        if key in self._dict:
            return self._dict[key]
        else:
            return None

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __str__(self):
        return self._dict.__str__()

