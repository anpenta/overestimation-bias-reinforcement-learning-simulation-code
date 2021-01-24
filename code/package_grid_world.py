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

# Package Grid World Module
# Model of a grid world where the agent must collect all the packages to end the episode.

import math

import numpy as np


class PackageGridWorld:

  def __init__(self, grid_dimension_size, reward_function):
    self._grid = np.zeros((grid_dimension_size, grid_dimension_size))
    self._cell_space = tuple([index for index, x in np.ndenumerate(self._grid)])
    self._action_space = ("left", "up", "right", "down", "collect")
    self._reward_function = reward_function

    # The starting cell is in the lower left corner and the package cells are along the walls of the grid.
    # Each package cell is associated with a boolean value that indicates whether the package is still
    # active in the world or not.
    self._starting_cell = (grid_dimension_size - 1, 0)
    self._package_cells = dict(zip(((0, 0), (0, grid_dimension_size - 1),
                                    (grid_dimension_size - 1, grid_dimension_size - 1),
                                    (0, math.ceil((grid_dimension_size - 1) / 2)),
                                    (grid_dimension_size - 1, math.ceil((grid_dimension_size - 1) / 2))),
                                   np.ones(5, dtype=bool)))

    self._agent_cell = self._starting_cell
    self._agent_action = None

  def _update_agent_cell(self):
    next_cell = None
    # Compute a next cell only if the agent decided to move.
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
    return 100 if self._is_state_terminal() else -1

  def _is_package_collected(self):
    return (self._agent_cell in self._package_cells.keys() and self._package_cells[self._agent_cell]
            and self._agent_action == "collect")

  def _is_state_terminal(self):
    return True not in self._package_cells.values()

  def _encode_state(self):
    # The agent's cell and the package cells are encoded to a two dimensional array,
    # before sent to the agent.
    agent_code = np.zeros(self._grid.shape[0] * self._grid.shape[1])
    agent_code[self._agent_cell[0] + self._agent_cell[1] * self._grid.shape[0]] = 1

    # For packages that are still active in the world, encode the value one in the appropriate
    # vector position. The rest of the elements of the package vector are zero.
    package_code = np.zeros(self._grid.shape[0] * self._grid.shape[1])
    for k, v in self._package_cells.items():
      package_code[k[0] + k[1] * self._grid.shape[0]] = int(v)

    state = np.stack((agent_code, package_code), axis=0)
    state = np.reshape(state, (1, *state.shape))

    return state

  def step(self, action):
    # In each step the environment reacts to the agent's action by returning an updated state encoding, a reward,
    # and a signal for determining whether the episode ended or not. The agent must collect all the packages to
    # end the episode.
    self._agent_action = self._action_space[action]
    if self._is_package_collected():
      self._package_cells[self._agent_cell] = False
    reward = self._generate_reward()
    done = self._is_state_terminal()
    if not done:
      self._update_agent_cell()
    return self._encode_state(), reward, done

  def reset(self):
    self._package_cells = dict(zip(self._package_cells.keys(), np.ones(len(self._package_cells), dtype=bool)))
    self._agent_cell = self._starting_cell
    self._agent_action = None
    return self._encode_state()

  def compute_action_space_size(self):
    return len(self._action_space)

  def compute_state_space_size(self):
    return 2 * len(self._cell_space)

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
  def package_cells(self):
    return self._package_cells

  @property
  def agent_action(self):
    return self._agent_action
