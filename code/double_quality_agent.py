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

# Double Quality Agent Module
# Model of an agent that uses Double Q-learning.

import math
import random

import numpy as np


class DoubleQualityAgent:

  def __init__(self, state_space_size, action_space_size):
    self._gamma = 0.95

    self._state_space_size = state_space_size
    self._action_space_size = action_space_size
    self._alpha_q_values = np.zeros((self._state_space_size, self._action_space_size))
    self._beta_q_values = np.zeros((self._state_space_size, self._action_space_size))
    self._mean_q_values = np.zeros((self._state_space_size, self._action_space_size))
    self._state_visit_count = np.zeros(self._state_space_size)
    self._alpha_q_value_update_count = np.zeros((self._state_space_size, self._action_space_size))
    self._beta_q_value_update_count = np.zeros((self._state_space_size, self._action_space_size))

  def determine_action(self, state):
    # Defining epsilon to be decreasing as the state visit count increases, ensures that all
    # the states will be visited enough times for the agent to explore, and that during the
    # end of learning the epsilon-greedy policy will asymptotically approach the greedy policy.
    self._state_visit_count[state] += 1
    epsilon = math.pow(self._state_visit_count[state], -0.5)

    if np.random.rand() <= epsilon:
      return np.random.randint(self._action_space_size)

    # In the greedy case, return the action that maximizes the mean Q values of the given state.
    return np.argmax(self._mean_q_values[state])

  def step(self, state, action, reward, next_state, done):
    # Update the alpha Q values or the beta Q values with equal probability. Use the Q function that is
    # going to be updated to select the argmax for the next state, and evaluate the argmax using the Q
    # function that is not going to be updated.
    # Defining alpha to be a decreasing polynomial function of the Q value update count, improves
    # the convergence rate.
    # If the episode ended, update the current state's Q value, using only the given reward as target.
    # Otherwise, use as target the sum of the given reward and the discounted maximum Q value of the next state.
    if random.getrandbits(1):
      self._alpha_q_value_update_count[state][action] += 1
      alpha = math.pow(self._alpha_q_value_update_count[state][action], -0.8)

      target = reward
      if not done:
        target_action = np.argmax(self._alpha_q_values[next_state])
        target += self._gamma * self._beta_q_values[next_state][target_action]

      self._alpha_q_values[state][action] += alpha * (target - self._alpha_q_values[state][action])
    else:
      self._beta_q_value_update_count[state][action] += 1
      alpha = math.pow(self._beta_q_value_update_count[state][action], -0.8)

      target = reward
      if not done:
        target_action = np.argmax(self._beta_q_values[next_state])
        target += self._gamma * self._alpha_q_values[next_state][target_action]

      self._beta_q_values[state][action] += alpha * (target - self._beta_q_values[state][action])

    # Update the mean Q value of the state-action pair that was updated.
    self._mean_q_values[state][action] = np.mean((self._alpha_q_values[state][action],
                                                  self._beta_q_values[state][action]))

  def compute_max_q_value(self, state):
    return max(np.max(self._alpha_q_values[state]), np.max(self._beta_q_values[state]))

  @property
  def gamma(self):
    return self._gamma
