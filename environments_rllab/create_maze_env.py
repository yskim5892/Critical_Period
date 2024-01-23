# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from environments_rllab.ant_maze_env import AntMazeEnv


def create_maze_env(env_name=None, scaling=8):
  maze_id = None
  if env_name.startswith('ant_reacher'):
    maze_id = 'Reacher'
  elif env_name.startswith('ant_maze_hard'):
    maze_id = 'MazeHard'
  elif env_name.startswith('ant_maze'):
    maze_id = 'Maze'
  elif env_name.startswith('ant_push'):
    maze_id = 'Push'
  elif env_name.startswith('ant_fall'):
    maze_id = 'Fall'
  elif env_name.startswith('ant_cross'):
    maze_id = 'Cross'
  else:
    raise ValueError('Unknown maze environment %s' % env_name)

  return AntMazeEnv(maze_id=maze_id, maze_size_scaling=scaling)
