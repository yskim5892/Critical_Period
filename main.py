import wandb
from general import utils
from general.configs import Configs
import config_presets
import datetime
import json
import sys
import os
import pdb
from environments import get_env
from agent import Agent

def create_dir(config):
    if config['date'] is None:
        date_string = str(datetime.date.today()).replace('-', '')
    else:
        date_string = str(config['date'])
    dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', '%s_%s'%(date_string, config['experiment_name']))
    config['dir_name'] = dir_name
    utils.create_muldir(dir_name)

    name_formats_file_path = os.path.join(dir_name, 'name_formats.json')
    if os.path.isfile(name_formats_file_path):
        if config['config_id'] is None:
            with open(name_formats_file_path, 'r') as f:
                name_formats = json.load(f)
    else:
        name_formats = config_presets.name_formats
        with open(name_formats_file_path, 'w') as f:
            json.dump(name_formats, f)
    if config['config_id'] is None:
        name_format = utils.NameFormat(name_formats['baseline'])
        config['config_id'] = name_format.get_id_from_config(config)

    if not config['load'] and config['run_id'] is None:
        for i in range(100):
            if not os.path.isdir(os.path.join(dir_name, config['config_id'], str(i))):
                config['run_id'] = i
                break
    if config['run_id'] is None:
        config['run_id'] = 0

    if config['gpu'] is None:
        config['gpu'] = config['run_id'] % 4

def train(agent, num_episodes):
    agent.is_testing = False
    agent.reset_stats()
    if num_episodes == 0:
        return
    for episode in range(num_episodes):
        phase = agent.train_steps // agent.config['curriculum_interval']
        phase = min(phase, len(agent.config['curriculum']) - 1)
        env.set_dense_level(int(agent.config['curriculum'][phase]) - 1)

        agent.run()
    agent.update_stats()

def test(agent, num_episodes):
    agent.is_testing = True
    successful_episodes = 0
    agent.reset_stats()
    if num_episodes == 0:
        return 0
    for episode in range(num_episodes):
        if agent.run():
            successful_episodes += 1
    agent.update_stats()
    return successful_episodes / num_episodes

def run_train_test(agent):
    config = agent.config
    if not config['no_record']:
        resume = "must" if config['wandb_resume'] else None
        wandb_id = config['wandb_id'] if config['wandb_id'] else None
        wandb.init(config=config._dict, entity=config['entity'], project=config['project'], name='%s/%s/%s'%(config['experiment_name'], config['config_id'], config['run_id']), resume=resume, id=wandb_id)
        agent.logger.log(f'Results at wandb.ai/%s/%s, run name %s/%s/%s'%(config['entity'], config['project'], config['experiment_name'], config['config_id'], config['run_id']))

    max_success_rate = -1
    for i in range(config['num_epochs']):
        epoch = i + config['start_epoch']
        train(agent, config['num_train_episodes'])

        if not config['no_record']:
            agent.record_stats(epoch, ['min_reward', 'max_reward', 'avg_reward', 'num_episodes', 'num_steps', 'num_successes', 'success_rate', 'avg_actor_loss', 'avg_critic_loss'])
            if config['algorithm'] == 'SAC':
                agent.record_stats(epoch, ['avg_alpha', 'avg_log_prob', 'avg_action_std'])
            elif config['algorithm'] == 'PPO':
                agent.record_stats(epoch, ['avg_ent_loss', 'avg_action_std'])

            if config['record_sharpness'] and agent.transition_sample.full():
                agent.actor.record_sharpness(agent.transition_sample)
                agent.record_stats(epoch, ['avg_grad_var', 'avg_normalized_grad_var', 'avg_hessian_norm', 'avg_hessian_frobenius_norm', 'avg_SAM_sharpness'])
        success_rate = test(agent, config['num_test_episodes'])
        if not config['no_record']:
            agent.record_stats(epoch, ['min_reward', 'max_reward', 'avg_reward', 'num_episodes', 'num_steps', 'num_successes', 'success_rate'])

        total_train_episodes = agent.total_train_episodes.numpy()
        
        log = f'Epoch {epoch} : Total Train Episodes {total_train_episodes} Success Rate {success_rate}'

        if config['env'] == 'cartpole':
            avg_reward = agent.stats['avg_reward']
            log += f' Avg Reward {avg_reward}'

        agent.logger.log(log)

        if config['save_every_n_episodes'] != -1 and total_train_episodes % config['save_every_n_episodes'] == 0:
            agent.save(save_by_episodes=True)

        if success_rate >= max_success_rate or success_rate == 1:
            max_success_rate = success_rate
            if config['save']:
                agent.save()

if __name__ == '__main__':
    config = Configs.load(preset_file_path='config_presets/default.json', argv=sys.argv)
    config_presets.postprocess_config(config)

    env = get_env(config)
    config['max_actions'] = env.max_actions
    config['timesteps_per_action'] = env.timesteps_per_action
    config['goal_thr'] = env.goal_thr
    create_dir(config)
    utils.select_gpu(config['gpu'])
    agent = Agent(config, env)

    run_train_test(agent)
    
