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

# Double Quality Network Agent Module
# Model of an agent that uses Double Q-learning with neural network function approximation.

import random

import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD


class DoubleQualityNetworkAgent:

  def __init__(self, state_space_size, action_space_size):
    self._gamma = 0.95
    self._max_epsilon = 1
    self._min_epsilon = 0.05
    self._epsilon_decay_steps = 750000
    self._epsilon_decay = (self._max_epsilon - self._min_epsilon) / self._epsilon_decay_steps
    self._optimizer = SGD(lr=0.005)

    self._state_space_size = state_space_size
    self._action_space_size = action_space_size
    self._alpha_model = self._build_model()
    self._beta_model = self._build_model()
    self._epsilon = self._max_epsilon
    self._model_choice = None

  def _build_model(self):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(self._action_space_size, activation="linear"))
    model.compile(loss="mse", optimizer=self._optimizer)
    return model

  def determine_action(self, state):
    # Decrease epsilon linearly to reduce the exploration gradually.
    if self._epsilon > self._min_epsilon:
      self._epsilon -= self._epsilon_decay

    if np.random.rand() <= self._epsilon:
      return np.random.randint(self._action_space_size)

    # In the greedy case, choose the alpha model or the beta model with equal probability and
    # return the action that maximizes the Q values of the given state.
    self._model_choice = "alpha" if random.getrandbits(1) else "beta"
    if self._model_choice == "alpha":
      return np.argmax(self._alpha_model.predict(state)[0])
    return np.argmax(self._beta_model.predict(state)[0])

  def step(self, state, action, reward, next_state, done):
    # Train the model that was used to select the action. Use the model that is going to be trained
    # to select the argmax for the next state, and evaluate the argmax using the model that is not
    # going to be trained.
    # The targets are the model's own predictions for actions that were not taken.
    # For the action that was taken, if the episode ended use only the given reward as target. Otherwise,
    # use as target the sum of the given reward and the discounted maximum Q value of the next state.
    if self._model_choice == "alpha":
      targets = self._alpha_model.predict(state)

      targets[0][action] = reward
      if not done:
        target_action = np.argmax(self._alpha_model.predict(next_state)[0])
        targets[0][action] += self._gamma * self._beta_model.predict(next_state)[0][target_action]

      self._alpha_model.fit(state, targets, batch_size=1, epochs=1, verbose=0)
    else:
      targets = self._beta_model.predict(state)

      targets[0][action] = reward
      if not done:
        target_action = np.argmax(self._beta_model.predict(next_state)[0])
        targets[0][action] += self._gamma * self._alpha_model.predict(next_state)[0][target_action]

      self._beta_model.fit(state, targets, batch_size=1, epochs=1, verbose=0)

  def compute_max_q_value(self, state):
    return max(np.max(self._alpha_model.predict(state)[0]), np.max(self._beta_model.predict(state)[0]))

  @property
  def gamma(self):
    return self._gamma
