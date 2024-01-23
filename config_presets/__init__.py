name_formats = {
    'baseline' : ['algorithm', 'curriculum', 'actor_lr', 'critic_lr', 'reward_setting', 'dense_reward_scale', 'HER', 'scaling', 'goal_thr', 'max_actions', 'timesteps_per_action']
}

lr_schedulers = {
    # sync_lr_decay : True
    'a1': {'initial_lr' : 1e-5, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    'a2': {'initial_lr' : 3e-5, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    'a3': {'initial_lr' : 1e-4, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    'a4': {'initial_lr' : 3e-4, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    'a5': {'initial_lr' : 1e-3, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    'a6': {'initial_lr' : 3e-3, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    'a7': {'initial_lr' : 1e-2, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    'a8': {'initial_lr' : 3e-2, 'decay_steps' : 2000, 'decay_rate' : 0.96},
    
    # sync_lr_decay : False
    'b1': {'initial_lr' : 1e-5, 'decay_steps' : 40000, 'decay_rate' : 0.96},
    'b2': {'initial_lr' : 3e-5, 'decay_steps' : 40000, 'decay_rate' : 0.96},
    'b3': {'initial_lr' : 1e-4, 'decay_steps' : 40000, 'decay_rate' : 0.96},
    'b4': {'initial_lr' : 3e-4, 'decay_steps' : 40000, 'decay_rate' : 0.96},
    'b5': {'initial_lr' : 1e-3, 'decay_steps' : 40000, 'decay_rate' : 0.96},
    'b6': {'initial_lr' : 3e-3, 'decay_steps' : 40000, 'decay_rate' : 0.96},
    'b7': {'initial_lr' : 1e-2, 'decay_steps' : 40000, 'decay_rate' : 0.96},
    'b8': {'initial_lr' : 3e-2, 'decay_steps' : 40000, 'decay_rate' : 0.96},
}

def get_lr_scheduler(lr_id):
    if lr_id in lr_schedulers:
        return lr_schedulers[lr_id]
    else:
        return {'initial_lr' : float(lr_id), 'decay_steps' : 10000, 'decay_rate' : 1}

def postprocess_config(config):
    config['use_gae'] = (config['gae_lambda'] > 0)
    
    config['curriculum'] = str(config['curriculum'])
    config['critic_lr_scheduler'] = get_lr_scheduler(config['critic_lr'])
    config['actor_lr_scheduler'] = get_lr_scheduler(config['actor_lr'])

    assert config['save_every_n_episodes'] == -1 or config['num_train_episodes'] % config['save_every_n_episodes'] == 0
        
