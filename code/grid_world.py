# Copyright (C) 2020 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Grid World Module
# Model of a grid world where the agent must reach the goal cell and perform an action to end the episode.

import random

import numpy as np


class GridWorld:

  def __init__(self, grid_dimension_size, reward_function, state_encoding):
    self._grid = np.zeros((grid_dimension_size, grid_dimension_size))
    self._cell_space = tuple([index for index, x in np.ndenumerate(self._grid)])
    self._action_space = ("left", "up", "right", "down")
    self._reward_function = reward_function
    self._state_encoding = state_encoding

    # The starting cell is in the lower left corner and the goal cell is in the upper right corner.
    self._starting_cell = (grid_dimension_size - 1, 0)
    self._goal_cell = (0, grid_dimension_size - 1)

    self._agent_cell = self._starting_cell
    self._agent_action = None

  def _update_agent_cell(self):
    if self._agent_action == "up":
      next_cell = (self._agent_cell[0] - 1, self._agent_cell[1])
    elif self._agent_action == "down":
      next_cell = (self._agent_cell[0] + 1, self._agent_cell[1])
    elif self._agent_action == "left":
      next_cell = (self._agent_cell[0], self._agent_cell[1] - 1)
    elif self._agent_action == "right":
      next_cell = (self._agent_cell[0], self._agent_cell[1] + 1)

    # Only update the cell if the new cell is valid. Else, assume that the cell remains the same.
    if next_cell in self._cell_space:
      self._agent_cell = next_cell

  def _generate_reward(self):
    if self._is_state_terminal():
      if self._reward_function == "bernoulli":
        return 50 if random.getrandbits(1) else -40
      else:
        return 5
    elif self._reward_function in ("non-terminal-bernoulli", "bernoulli"):
      return 10 if random.getrandbits(1) else -12
    elif self._reward_function == "high-variance-gaussian":
      return np.random.normal(-1, 5)
    elif self._reward_function == "low-variance-gaussian":
      return np.random.normal(-1, 1)

  def _is_state_terminal(self):
    return self._agent_cell == self._goal_cell

  def _encode_state(self):
    # If the state encoding is integer, the agent's cell is encoded to an integer before sent to the agent. If the
    # state encoding is array, the agent's cell and the goal cell are encoded to a two dimensional array before sent
    # to the agent.
    if self._state_encoding == "integer":
      state = self._agent_cell[0] + self._agent_cell[1] * self._grid.shape[0]
    elif self._state_encoding == "array":
      agent_code = np.zeros(self._grid.shape[0] * self._grid.shape[1])
      agent_code[self._agent_cell[0] + self._agent_cell[1] * self._grid.shape[0]] = 1

      goal_code = np.zeros(self._grid.shape[0] * self._grid.shape[1])
      goal_code[self._goal_cell[0] + self._goal_cell[1] * self._grid.shape[0]] = 1

      state = np.stack((agent_code, goal_code), axis=0)
      state = np.reshape(state, (1, *state.shape))

    return state

  def step(self, action):
    # In each step the environment reacts to the agent's action by returning an updated state encoding, a reward,
    # and a signal for determining whether the episode ended or not. The agent must take an action in the
    # goal cell to end the episode.
    self._agent_action = self._action_space[action]
    reward = self._generate_reward()
    done = self._is_state_terminal()
    if not done:
      self._update_agent_cell()
    return self._encode_state(), reward, done

  def reset(self):
    self._agent_cell = self._starting_cell
    self._agent_action = None
    return self._encode_state()

  def compute_action_space_size(self):
    return len(self._action_space)

  def compute_state_space_size(self):
    cell_space_length = len(self._cell_space)
    if self._state_encoding == "integer":
      return cell_space_length
    elif self._state_encoding == "array":
      return 2 * cell_space_length

  @property
  def reward_function(self):
    return self._reward_function

  @property
  def grid(self):
    return self._grid

  @property
  def agent_cell(self):
    return self._agent_cell

  @property
  def goal_cell(self):
    return self._goal_cell

  @property
  def agent_action(self):
    return self._agent_action
