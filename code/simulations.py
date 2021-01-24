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

# Simulations Module
# Reinforcement learning simulation functions.
# In every simulation function we start counting timesteps from 0.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utility

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["font.size"] = 12


# simulate_visual_test_episode: Simulates one test episode with visual effects. The given agent interacts
# with the given environment without updating its Q values, until the episode ends or the given number of
# maximum timesteps is exceeded.
def simulate_visual_test_episode(agent, environment, max_timesteps=40, verbose=False):
  utility.print_line()
  print("Simulating a visual test episode for a maximum of {} timestep(s)".format(max_timesteps))
  utility.print_line()

  plt.ion()

  total_reward = 0
  state = environment.reset()
  for t in range(max_timesteps):

    if verbose:
      print("Timestep: {:>4}".format(t), sep=" ", end="", flush=True)
      print(" | Agent's cell: {}".format(environment.agent_cell), sep=" ", end="", flush=True)

    utility.visualize_grid(agent, environment)
    plt.pause(2.5)

    action = agent.determine_action(state)
    next_state, reward, done = environment.step(action)

    if verbose:
      print(" | Action taken: {:>7}".format(environment.agent_action), sep=" ", end="", flush=True)
      print(" | Reward given: {:>3}".format(reward))

    total_reward += reward
    state = next_state

    if done:
      break

  print("Total timesteps: {}".format(t + 1), sep=" ", end="", flush=True)
  print(" | Total reward gained: {}".format(total_reward), sep=" ", end="", flush=True)
  print(" | Episode ended: {}".format(done))
  print()

  plt.ioff()
  plt.close()


# simulate_training_episodes: Simulates the given number of training episodes. In each episode, the given
# agent interacts with the given environment and updates its Q values, until the episode ends or the given
# number of maximum timesteps is exceeded.
def simulate_training_episodes(agent, environment, episodes=10000, max_timesteps=1000, visual_evaluation_frequency=0,
                               evaluate_trained_agent=False):
  utility.print_line()
  print("Simulating {} training episode(s) for a maximum of {} timestep(s) each".format(episodes, max_timesteps))
  utility.print_line()

  for i in range(episodes):
    total_reward = 0
    state = environment.reset()
    for t in range(max_timesteps):
      action = agent.determine_action(state)
      next_state, reward, done = environment.step(action)
      agent.step(state, action, reward, next_state, done)

      total_reward += reward
      state = next_state

      if done:
        break

    print("Episode: {:>5}".format(i + 1), sep=" ", end="", flush=True)
    print(" | Total timesteps: {:>4}".format(t + 1), sep=" ", end="", flush=True)
    print(" | Total reward gained: {:>5}".format(total_reward), sep=" ", end="", flush=True)
    print(" | Episode ended: {}".format(done))

    if visual_evaluation_frequency and (i + 1) % visual_evaluation_frequency == 0:
      print()
      print("Visually evaluating agent after episode {}".format(i + 1))
      simulate_visual_test_episode(agent, environment, verbose=True)

  if evaluate_trained_agent:
    print()
    print("Evaluating trained agent")
    simulate_test_episodes(agent, environment)


# simulate_test_episodes: Simulates the given number of test episodes. In each episode, the given
# agent interacts with the given environment without updating its Q values, until the episode ends.
# The mean total reward is computed over all episodes and the mean discounted return is computed
# over all timesteps. The output is printed in the console.
def simulate_test_episodes(agent, environment, episodes=10000):
  utility.print_line()
  print("Simulating {} test episode(s)".format(episodes))
  utility.print_line()

  gamma = agent.gamma
  total_rewards = []
  discounted_returns = []
  for i in range(episodes):
    total_reward = 0
    discounted_return_coefficient = 0
    state = environment.reset()
    done = False
    t = 0
    while not done:
      action = agent.determine_action(state)
      next_state, reward, done = environment.step(action)

      total_reward += reward
      discounted_return_coefficient += gamma ** t
      discounted_returns.append(reward * discounted_return_coefficient)
      state = next_state

      t += 1

    total_rewards.append(total_reward)

  # Print the data in the console.
  environment_type = utility.determine_environment_type(environment)
  agent_type = utility.determine_agent_type(agent)
  grid_dimension_size = environment.grid.shape[0]

  print("Total timesteps: {}".format(len(discounted_returns)), sep=" ", end="", flush=True)
  print(" | Agent type: {}".format(agent_type), sep=" ", end="", flush=True)
  print(" | Environment type: {}".format(environment_type), sep=" ", end="", flush=True)
  print(" | Grid dimension size: {}".format(grid_dimension_size), sep=" ", end="", flush=True)
  print(" | Discount factor: {}".format(gamma))

  print("Mean total reward over all episodes: {}".format(np.mean(total_rewards)))
  print("Mean discounted return over all timesteps: {}".format(np.mean(discounted_returns)))


# simulate_training_timesteps: Simulates the given number of training timesteps. At each timestep, the given
# agent interacts with the given environment and updates its Q values. The reward, the state max Q value, and
# the starting state max Q value are collected at each timestep, and stored in the given output path.
def simulate_training_timesteps(agent, environment, timesteps=10000, verbose=False, output_path="./output"):
  utility.print_line()
  print("Simulating {} training timestep(s)".format(timesteps))
  utility.print_line()

  rewards = []
  max_q_values = []
  starting_state_max_q_values = []
  starting_state = environment.reset()
  state = environment.reset()
  for t in range(timesteps):
    if verbose:
      print("Simulating timestep: {:>5}/{}".format(t, timesteps))

    action = agent.determine_action(state)
    next_state, reward, done = environment.step(action)
    agent.step(state, action, reward, next_state, done)

    rewards.append(reward)
    max_q_values.append(agent.compute_max_q_value(state))
    starting_state_max_q_values.append(agent.compute_max_q_value(starting_state))

    state = next_state

    if done:
      state = environment.reset()

  # Create dataframes and store the data in the given output path.
  environment_type = utility.determine_environment_type(environment)
  agent_type = utility.determine_agent_type(agent)
  grid_dimension_size = environment.grid.shape[0]

  data = pd.DataFrame()
  data["reward"] = rewards
  data["max_q_value"] = max_q_values
  data["starting_state_max_q_value"] = starting_state_max_q_values

  meta_data = pd.DataFrame()
  meta_data["timesteps"] = [timesteps]
  meta_data["agent_type"] = [agent_type]
  meta_data["environment_type"] = [environment_type]
  meta_data["grid_dimension_size"] = [grid_dimension_size]

  utility.save_dataframe(data, output_path, utility.join_strings("-", "training-timesteps", agent_type, environment_type))
  utility.save_dataframe(meta_data, output_path, utility.join_strings("-", "training-timesteps", agent_type,
                                                                      environment_type, "meta"))


# simulate_training_experiments: Simulates the given number of training experiments. In each experiment,
# a new agent of the given agent type interacts with the given environment and updates its Q values,
# until the given number of timesteps is exceeded. At each timestep, the mean reward, the mean starting
# state max Q value, and the mean normalized entropy of the state visits are computed incrementally over
# all the experiments, and stored in the given output path.
def simulate_training_experiments(agent_type, environment, experiments=10000, timesteps=10000, verbose=False,
                                  output_path="./output"):
  utility.print_line()
  print("Simulating {} training experiment(s) of {} timestep(s) each".format(experiments, timesteps))
  utility.print_line()

  state_space_size = environment.compute_state_space_size()
  action_space_size = environment.compute_action_space_size()
  mean_rewards = np.zeros(timesteps)
  mean_starting_state_max_q_values = np.zeros(timesteps)
  mean_normalized_entropies = np.zeros(timesteps)
  starting_state = environment.reset()
  for i in range(experiments):
    if verbose:
      print("Simulating experiment: {:>5}/{}".format(i + 1, experiments))

    state_visits = np.zeros(state_space_size)
    agent = utility.create_agent(agent_type, state_space_size, action_space_size)
    state = environment.reset()
    for t in range(timesteps):
      action = agent.determine_action(state)
      next_state, reward, done = environment.step(action)
      agent.step(state, action, reward, next_state, done)

      # For the current timestep, compute the incremental means of the rewards, the
      # starting state max q values, and the normalized entropies over the current experiment.
      mean_rewards[t] += (reward - mean_rewards[t]) / (i + 1)
      mean_starting_state_max_q_values[t] += (agent.compute_max_q_value(starting_state)
                                              - mean_starting_state_max_q_values[t]) / (i + 1)
      state_visits[next_state] += 1
      probabilities = state_visits / np.sum(state_visits)
      normalized_entropy = -np.nansum(probabilities * np.log2(probabilities)) / np.log2(state_space_size)
      mean_normalized_entropies[t] += (normalized_entropy - mean_normalized_entropies[t]) / (i + 1)

      state = next_state

      if done:
        state = environment.reset()

  # Create dataframes and store the data in the given output path.
  environment_type = utility.determine_environment_type(environment)

  data = pd.DataFrame()
  data["mean_reward"] = mean_rewards
  data["mean_starting_state_max_q_value"] = mean_starting_state_max_q_values
  data["mean_normalized_entropy"] = mean_normalized_entropies

  meta_data = pd.DataFrame()
  meta_data["timesteps"] = [timesteps]
  meta_data["experiments"] = [experiments]
  meta_data["agent_type"] = [agent_type]
  meta_data["environment_type"] = [environment_type]
  meta_data["grid_dimension_size"] = [environment.grid.shape[0]]

  utility.save_dataframe(data, output_path, utility.join_strings("-", "training-experiments", agent_type, environment_type))
  utility.save_dataframe(meta_data, output_path, utility.join_strings("-", "training-experiments", agent_type,
                                                                      environment_type, "meta"))
