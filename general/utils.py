import os
import glob
import time
import numpy as np
import tensorflow as tf
from math import sqrt
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform


class Logger:
    colors = {
        'black' : 30,
        'red' : 31,
        'green' : 32,
        'yellow' : 33,
        'blue' : 34,
        'magenta' : 35,
        'cyan' : 36,
        'white' : 37,
        'end' : 0
    }

    def __init__(self, log_dir, also_print=True, enable_time_stamp=False):
        self.log_path = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S.log", time.gmtime()))
        self.also_print = also_print
        self.record_time_stamp = enable_time_stamp
        self.start_time = 0

    def reset_time_stamp(self):
        self.start_time = 0

    def time_stamp(self, msg, tab=0):
        if not self.record_time_stamp:
            return
        for _ in tab:
            msg = '\t' + msg
        msg = '%.4f'%(time.time() - self.start_time) + msg
        self.log(msg)

    def log(self, *msgs):
        timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        msg = '[' + timestr + ']'
        msg_log = msg
        for m in msgs:
            if type(m) is tuple:
                for option in m:
                    num = 0
                    if option == 'underline':
                        num = 4
                    else:
                        if option.startswith('bg'):
                            option = option[2:]
                            num += 10
                        if option.startswith('bright'):
                            option = option[6:]
                            num += 60
                        if option in self.colors:
                            num += self.colors[option]
                    msg += f'\033[{num}m'
            else:
                msg += ' ' + str(m)
                msg_log += ' ' + str(m)
        msg += f'\033[0m'
        
        if self.also_print:
            print(msg)

        with open(self.log_path, 'a') as f:
            f.write(msg_log + '\n')

def params2id(*args):
    nargs = len(args)
    id_ = '{}'+'_{}'*(nargs-1)
    return id_.format(*args)

class NameFormat:
    def __init__(self, attrs):
        self.attrs = attrs
        self.nattr = len(self.attrs)
        for attr in self.attrs:
            assert type(attr)==str, "Type of attributes should be sting"

    def get_id_from_config(self, config):
        params = []
        for attr in self.attrs:
            param = config[attr]
            if type(param) == float:
                param = round(param, 5)
            params.append(param)
        return params2id(*tuple(params))

    def get_query_file_id_from_config(self, config, invariables):
        return params2id(*tuple(['*' if not attr in invariables else config[attr] for attr in self.attrs]))

    def get_config_names_indices(self, config_names):
        return [self.attrs.index(arg_name) for config_name in arg_names]

    def update_config_with_id(self, config, id_):
        id_split = id_.split('_')
        assert len(id_split)==self.nattr, "The number of components of id_ and the number of attributes should be same"

        for i in range(self.nattr):
            attr = self.attrs[i]
            type_attr = type(config[attr])
            config[attr] = type_attr(id_split[i])

def create_dir(dirname):
    if not os.path.exists(dirname):
        print("Creating %s"%dirname)
        os.makedirs(dirname)
    else:
        pass

def create_muldir(*args):
    for dirname in args:
        create_dir(dirname)

def add_dict_by_step(dict_, params_, values_by_step):
    if params_ not in dict_:
        dict_[params_] = {} # {step : [mean, square_mean, n]}

    for step in values_by_step:
        if step not in dict_[params_]:
            v = values_by_step[step]
            dict_[params_][step] = [v, v**2, 1]
        else:
            old_mean = dict_[params_][step][0]
            old_sq_mean = dict_[params_][step][1]
            n = dict_[params_][step][2]
            v = values_by_step[step]
            dict_[params_][step][0] = (old_mean * n + v) / (n + 1)
            dict_[params_][step][1] = (old_sq_mean * n + v**2) / (n + 1)
            dict_[params_][step][2] = n + 1

def add_dict(dict_, params_, value):
    if params_ not in dict_:
        dict_[params_] = []
    dict_[params_].append(value)

def append_queue_mean(step, queue_, dict_, val_):
    if queue_.full():
        queue_.get()
    queue_.put(val_)
    dict_[step] = np.mean(list(queue_.queue))

def select_gpu(id_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(id_)

def lr_decayed(scheduler):
    return tf.keras.optimizers.schedules.ExponentialDecay(scheduler['initial_lr'], scheduler['decay_steps'], scheduler['decay_rate'], staircase=True)

class SummaryWriter:
    def __init__(self, save_path):
        self.writer = tf.summary.FileWriter(save_path)
        create_dir(save_path)

    def add_summary(self, tag, simple_value, global_step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)])
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, dict_, global_step):
        for key in dict_.keys():
            self.add_summary(str(key), dict_[key], global_step)


def unique_filename(basename, ext):
    name = f'{basename}.{ext}'
    i = 1
    while os.path.exists(name):
        name = f'{basename}{i}.{ext}'
        i += 1
    return name

def process_input(arr, logger):
    start_time = time.time()
    logger.log('%.4f\t\t\t\tStart'%(time.time() - start_time))
    if arr is None:
        return None
    logger.log('%.4f\t\t\t\tArr to Numpy'%(time.time() - start_time))
    if not tf.is_tensor(arr):
        arr = np.array(arr, dtype='float32')
    logger.log('%.4f\t\t\t\tExpand Dims'%(time.time() - start_time))
    if len(arr.shape) == 1:
        arr = tf.expand_dims(arr, 0)
    logger.log('%.4f\t\t\t\tEnd'%(time.time() - start_time))

    return arr

def create_layers(num_neurons, name_format, activation='relu', initializer=None):
    layers = []
    for i in range(len(num_neurons)):
        n = num_neurons[i]
        if initializer is None:
            layers.append(Dense(n, activation=activation, name = name_format%i, kernel_initializer=RandomUniform(-1/sqrt(n), 1/sqrt(n))))
        else:
            layers.append(Dense(n, activation=activation, name = name_format%i, kernel_initializer = initializer))
    return layers

def softmax(x, axis=-1):
    y = np.exp(x - np.max(x))
    return y / np.sum(y, axis=axis)

def accuracy_with_threshold(pred, label, thr):
    if tf.is_tensor(pred):
        pred = pred.numpy()
    neg_pred = np.where(np.squeeze(pred) <= thr, -1, 0)
    pos_pred = np.where(np.squeeze(pred) >= 1 - thr, 1, 0)
    pred_discrete = (neg_pred + pos_pred + 1) / 2

    accuracy = np.sum(np.equal(pred_discrete, label)) / len(label)
    return accuracy

class BoxSpace:
    # N-dimensional Box-shape space defined by bounds, or extents and center
    # bounds.shape : n * 2
    # center.shape : n
    # extents.shape : n

    def __init__(self, bounds = None, center = None, extents = None):
        if bounds is not None:
            self.bounds = np.array(bounds)
            self.extents = (self.bounds[:, 1] - self.bounds[:, 0]) / 2
            self.center = self.bounds[:, 1] - self.extents
        else: # Assume extents and center is not None
            self.extents = np.array(extents)
            self.center = np.array(center)
            self.bounds = np.stack([self.center - self.extents, self.center + self.extents], axis=1)
        self.dim = len(self.center)

    def random_sample(self):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def clip(self, p, dims=None):
        if dims is not None:
            bounds = np.stack([np.ones(self.dim) * float('-inf'), np.ones(self.dim) * float('inf')], 1)
            bounds[dims] = self.bounds[dims]
        else:
            bounds = self.bounds
        return np.clip(p, bounds[:, 0], bounds[:, 1])
