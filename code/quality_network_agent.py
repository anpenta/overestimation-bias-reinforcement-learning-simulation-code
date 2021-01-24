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

# Quality Network Agent Module
# Model of an agent that uses Q-learning with neural network function approximation.

import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD


class QualityNetworkAgent:

  def __init__(self, state_space_size, action_space_size):
    self._gamma = 0.95
    self._max_epsilon = 1
    self._min_epsilon = 0.05
    self._epsilon_decay_steps = 750000
    self._epsilon_decay = (self._max_epsilon - self._min_epsilon) / self._epsilon_decay_steps
    self._optimizer = SGD(lr=0.005)

    self._state_space_size = state_space_size
    self._action_space_size = action_space_size
    self._model = self._build_model()
    self._epsilon = self._max_epsilon

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

    return np.argmax(self._model.predict(state)[0])

  def step(self, state, action, reward, next_state, done):
    # The targets are the model's own predictions for actions that were not taken.
    targets = self._model.predict(state)

    # For the action that was taken, if the episode ended use only the given reward as target. Otherwise,
    # use as target the sum of the given reward and the discounted maximum Q value of the next state.
    targets[0][action] = reward
    if not done:
      targets[0][action] += self._gamma * np.max(self._model.predict(next_state)[0])

    self._model.fit(state, targets, batch_size=1, epochs=1, verbose=0)

  def compute_max_q_value(self, state):
    return np.max(self._model.predict(state)[0])

  @property
  def gamma(self):
    return self._gamma
