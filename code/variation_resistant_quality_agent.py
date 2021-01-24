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

# Variation-resistant Quality Agent Module
# Model of an agent that uses Variation-resistant Q-learning.

import collections
import math

import numpy as np


class VariationResistantQualityAgent:

  def __init__(self, state_space_size, action_space_size):
    self._gamma = 0.95
    self._q_value_memory_capacity = 150
    self._lambda = -3

    self._state_space_size = state_space_size
    self._action_space_size = action_space_size
    self._q_values = np.zeros((self._state_space_size, self._action_space_size))
    self._state_visit_count = np.zeros(self._state_space_size)
    self._q_value_update_count = np.zeros((self._state_space_size, self._action_space_size))
    self._q_value_memory = {}

  def _store_q_value(self, state, action):
    if (state, action) not in self._q_value_memory:
      self._q_value_memory[(state, action)] = collections.deque(maxlen=self._q_value_memory_capacity)
    self._q_value_memory[(state, action)].append(self._q_values[state][action])

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
    # and the discounted translated maximum Q value of the next state.
    target = reward
    if not done:
      target_action = np.argmax(self._q_values[next_state])
      if (next_state, target_action) not in self._q_value_memory:
        mean_absolute_deviation = 0
      else:
        mean = np.mean(self._q_value_memory[(next_state, target_action)])
        mean_absolute_deviation = np.mean(np.abs(self._q_value_memory[(next_state, target_action)] - mean))
      target_q_value = self._q_values[next_state][target_action] + self._lambda * mean_absolute_deviation
      target += self._gamma * target_q_value

    # Perform the update and store the new Q value estimate in memory.
    self._q_values[state][action] += alpha * (target - self._q_values[state][action])
    self._store_q_value(state, action)

  def compute_max_q_value(self, state):
    return np.max(self._q_values[state])

  @property
  def gamma(self):
    return self._gamma
