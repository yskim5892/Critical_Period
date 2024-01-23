from .ant_four_rooms import AntFourRoomsEnvironment
from .ur5 import UR5Environment
from .pendulum import PendulumEnvironment
from .ant_reacher import AntReacherEnvironment
from .ant_maze import AntMazeEnvironment
from .ant_push import AntPushEnvironment
from .ant_fall import AntFallEnvironment
from .ant_cross import AntCrossEnvironment
from .cartpole import CartPoleEnvironment
from .reacher import ReacherEnvironment

def get_env(config):
    name = config['env']
    config['xml_filename'] = name
    if name == 'ur5':
        return UR5Environment(config)
    if name == 'pendulum':
        return PendulumEnvironment(config)
    if name == 'ant_reacher':
        return AntReacherEnvironment(config)
    if name == 'ant_four_rooms':
        return AntFourRoomsEnvironment(config)
    if name == 'ant_maze':
        return AntMazeEnvironment(config)
    if name == 'ant_maze_hard':
        return AntMazeEnvironment(config)
    if name == 'ant_push':
        return AntPushEnvironment(config)
    if name == 'ant_fall':
        return AntFallEnvironment(config)
    if name == 'ant_cross':
        return AntCrossEnvironment(config)
    if name == 'cartpole':
        return CartPoleEnvironment(config)
    if name == 'reacher':
        return ReacherEnvironment(config)
