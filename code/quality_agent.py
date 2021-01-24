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

# Quality Agent Module
# Model of an agent that uses Q-learning.

import math

import numpy as np


class QualityAgent:

  def __init__(self, state_space_size, action_space_size):
    self._gamma = 0.95

    self._state_space_size = state_space_size
    self._action_space_size = action_space_size
    self._q_values = np.zeros((self._state_space_size, self._action_space_size))
    self._state_visit_count = np.zeros(self._state_space_size)
    self._q_value_update_count = np.zeros((self._state_space_size, self._action_space_size))

  def determine_action(self, state):
    # Defining epsilon to be decreasing as the state visit count increases, ensures that all
    # the states will be visited enough times for the agent to explore, and that during the
    # end of learning the epsilon-greedy policy will asymptotically approach the greedy policy.
    self._state_visit_count[state] += 1
    epsilon = math.pow(self._state_visit_count[state], -0.5)

    if np.random.rand() <= epsilon:
      return np.random.randint(self._action_space_size)

    return np.argmax(self._q_values[state])

  def step(self, state, action, reward, next_state, done):
    # Defining alpha to be a decreasing polynomial function of the Q value update count, improves
    # the convergence rate.
    self._q_value_update_count[state][action] += 1
    alpha = math.pow(self._q_value_update_count[state][action], -0.8)

    # If the episode ended, use only the given reward as target. Otherwise, use as target the sum of the given reward
    # and the discounted maximum Q value of the next state.
    target = reward
    if not done:
      target += self._gamma * np.max(self._q_values[next_state])

    # Perform the update.
    self._q_values[state][action] += alpha * (target - self._q_values[state][action])

  def compute_max_q_value(self, state):
    return np.max(self._q_values[state])

  @property
  def gamma(self):
    return self._gamma
