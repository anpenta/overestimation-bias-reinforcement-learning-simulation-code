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

# Utility Module
# Utility functions to run reinforcement learning simulations.

import argparse
import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K

import double_quality_agent
import double_quality_network_agent
import grid_world
import package_grid_world
import quality_agent
import quality_network_agent
import variation_resistant_quality_agent
import variation_resistant_quality_network_agent

plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["font.size"] = 12


def create_agent(agent_type, state_space, action_space):
  if agent_type == "quality-agent":
    return quality_agent.QualityAgent(state_space, action_space)
  elif agent_type == "double-quality-agent":
    return double_quality_agent.DoubleQualityAgent(state_space, action_space)
  elif agent_type == "variation-resistant-quality-agent":
    return variation_resistant_quality_agent.VariationResistantQualityAgent(state_space, action_space)
  elif agent_type == "quality-network-agent":
    return quality_network_agent.QualityNetworkAgent(state_space, action_space)
  elif agent_type == "double-quality-network-agent":
    return double_quality_network_agent.DoubleQualityNetworkAgent(state_space, action_space)
  elif agent_type == "variation-resistant-quality-network-agent":
    return variation_resistant_quality_network_agent.VariationResistantQualityNetworkAgent(state_space, action_space)


def create_environment(environment_type, grid_dimension_size, reward_function, state_encoding):
  if environment_type == "grid-world":
    return grid_world.GridWorld(grid_dimension_size, reward_function, state_encoding)
  elif environment_type == "package-grid-world":
    return package_grid_world.PackageGridWorld(grid_dimension_size, reward_function)


def determine_agent_type(agent):
  if type(agent) == quality_agent.QualityAgent:
    return "quality-agent"
  elif type(agent) == double_quality_agent.DoubleQualityAgent:
    return "double-quality-agent"
  elif type(agent) == variation_resistant_quality_agent.VariationResistantQualityAgent:
    return "variation-resistant-quality-agent"
  elif type(agent) == quality_network_agent.QualityNetworkAgent:
    return "quality-network-agent"
  elif type(agent) == double_quality_network_agent.DoubleQualityNetworkAgent:
    return "double-quality-network-agent"
  elif type(agent) == variation_resistant_quality_network_agent.VariationResistantQualityNetworkAgent:
    return "variation-resistant-quality-network-agent"


def determine_environment_type(environment):
  if type(environment) == grid_world.GridWorld:
    return join_strings("-", environment.reward_function, "grid-world")
  elif type(environment) == package_grid_world.PackageGridWorld:
    return join_strings("-", environment.reward_function, "package-grid-world")


def save_dataframe(dataframe, directory_path, basename):
  if not os.path.isdir(directory_path):
    print("Output directory does not exist | Creating directories along directory path")
    os.makedirs(directory_path)

  filename = basename + ".csv"
  print("Saving data | Filename: {} | Directory path: {}".format(filename, directory_path))
  dataframe.to_csv("{}/{}".format(directory_path, filename), index=False)


def save_model(model, directory_path, basename):
  if not os.path.isdir(directory_path):
    print("Output directory does not exist | Creating directories along directory path")
    os.makedirs(directory_path)

  filename = basename + ".h5"
  print("Saving model | Filename: {} | Directory path: {}".format(filename, directory_path))
  model.save("{}/{}".format(directory_path, filename))


def compute_incremental_mean(mean, count, new_value):
  count += 1
  mean += (new_value - mean) / count
  return mean, count


def compute_incremental_statistics(sum_of_squares, mean, count, new_value):
  # Update the mean and sum of squares incrementally using Welford's algorithm.
  count += 1
  delta = new_value - mean
  mean += delta / count
  sum_of_squares += delta * (new_value - mean)
  return sum_of_squares, mean, count


def compute_starting_state_max_optimal_q_value(environment_type, grid_dimension_size, gamma):
  if environment_type == "package-grid-world" and grid_dimension_size == 10:
    discounted_return = 0
    for t in range(32):
      reward = 100 if t == 31 else -1
      discounted_return += reward * gamma ** t
    return discounted_return
  elif environment_type == "grid-world":
    discounted_return = 5 * gamma ** (2 * (grid_dimension_size - 1))
    for t in range(2 * grid_dimension_size - 2):
      discounted_return -= gamma ** t
    return discounted_return


def compute_optimal_expected_discounted_return_per_timestep(environment_type, grid_dimension_size, gamma):
  if environment_type == "package-grid-world" and grid_dimension_size == 10:
    discounted_returns = []
    discounted_return_coefficient = 0
    for t in range(32):
      discounted_return_coefficient += gamma ** t
      reward = 100 if t == 31 else -1
      discounted_returns.append(reward * discounted_return_coefficient)
    return np.mean(discounted_returns)
  elif environment_type == "grid-world":
    discounted_returns = []
    discounted_return_coefficient = 0
    for t in range(2 * (grid_dimension_size - 1) + 1):
      discounted_return_coefficient += gamma ** t
      reward = 5 if t == 2 * (grid_dimension_size - 1) else -1
      discounted_returns.append(reward * discounted_return_coefficient)
    return np.mean(discounted_returns)


def print_line():
  print("-" * 100)


def visualize_grid(agent, environment):
  # Extract information from the agent and the environment.
  agent_type = determine_agent_type(agent)
  environment_type = determine_environment_type(environment)
  grid = environment.grid
  agent_cell = environment.agent_cell
  if "package-grid-world" in environment_type:
    package_cells = environment.package_cells
  else:
    goal_cell = environment.goal_cell

  # Create the visualization.
  plt.grid(True)
  ax = plt.gca()
  ax.set_title(format_for_plot(environment_type))
  ax.set_xticks(np.arange(0.5, grid.shape[0], 1))
  ax.set_yticks(np.arange(0.5, grid.shape[1], 1))
  ax.set_xticklabels([])
  ax.set_yticklabels([])

  canvas = np.copy(grid)
  agent_patch = mpatches.Patch(color="gray", label=format_for_plot(agent_type))
  canvas[agent_cell] = 5
  if "package-grid-world" in environment_type:
    active_package_patch = mpatches.Patch(color="white", label="Active Package")
    ax.legend(handles=[agent_patch, active_package_patch], facecolor="blue", bbox_to_anchor=(0.75, 0.015))
    for k, v in package_cells.items():
      if v:
        canvas[k] = 10
  else:
    goal_patch = mpatches.Patch(color="white", label="Goal")
    ax.legend(handles=[agent_patch, goal_patch], facecolor="blue", bbox_to_anchor=(0.75, 0.015))
    canvas[goal_cell] = 10

  plt.imshow(canvas, cmap="gray")


def format_for_plot(type_string):
  return type_string.replace("-", " ").title()


def join_strings(separator, *strings):
  return separator.join(strings)


def plot_total_rewards(data, agent_type, environment_type):
  plt.plot(data)
  plt.xlabel("Episode")
  plt.ylabel("Total reward")
  plt.title(join_strings(" in\n", format_for_plot(agent_type), format_for_plot(environment_type)))
  plt.show()


def control_randomness(seed):
  # Make sure the results are reproducible.
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  np.random.seed(seed)
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)

  configuration = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  tf.set_random_seed(seed)
  session = tf.Session(graph=tf.get_default_graph(), config=configuration)
  K.set_session(session)


def handle_input_argument_errors(input_arguments):
  if "network" not in input_arguments.agent_type and not input_arguments.environment_type == "grid-world":
    raise ValueError("value of agent_type is not a network agent and value of environment_type is not grid-world")
  elif not input_arguments.reward_function == "deterministic" and not input_arguments.environment_type == "grid-world":
    raise ValueError("value of reward_function is not deterministic and value of environment_type is not grid-world")
  elif input_arguments.reward_function == "deterministic" and input_arguments.environment_type == "grid-world":
    raise ValueError("value of reward_function is deterministic and value of environment_type is grid-world")
  elif input_arguments.simulation_function == "training_episodes":
    if input_arguments.episodes < input_arguments.visual_evaluation_frequency:
      raise ValueError("value of visual_evaluation_frequency is greater than value of episodes")
    elif input_arguments.visual_evaluation_frequency < 0:
      raise ValueError("value of visual_evaluation_frequency is negative")


def parse_input_arguments(agent_type_choices=("quality-agent", "double-quality-agent",
                                              "variation-resistant-quality-agent",
                                              "quality-network-agent", "double-quality-network-agent",
                                              "variation-resistant-quality-network-agent"),
                          environment_type_choices=("grid-world", "package-grid-world"),
                          reward_function_choices=("bernoulli", "non-terminal-bernoulli", "high-variance-gaussian",
                                                   "low-variance-gaussian", "deterministic"),
                          grid_dimension_size_choices=range(3, 11), episode_choices=range(1000, 10001, 1000),
                          max_timestep_choices=range(500, 10001, 500), seed_choices=range(1, 31, 1),
                          experiment_choices=range(1000, 10001, 1000), timestep_choices=range(10000, 1000001, 10000)):
  parser = argparse.ArgumentParser(prog="simulate", usage="runs reinforcement learning simulations")
  subparsers = parser.add_subparsers(dest="simulation_function", help="simulation function to run")
  subparsers.required = True

  add_training_episodes_parser(subparsers, agent_type_choices, environment_type_choices, reward_function_choices,
                               grid_dimension_size_choices, episode_choices, max_timestep_choices, seed_choices)
  add_training_experiments_parser(subparsers, agent_type_choices, environment_type_choices, reward_function_choices,
                                  grid_dimension_size_choices, experiment_choices, timestep_choices)
  add_training_timesteps_parser(subparsers, agent_type_choices, environment_type_choices, reward_function_choices,
                                grid_dimension_size_choices, timestep_choices, seed_choices)

  input_arguments = parser.parse_args()
  handle_input_argument_errors(input_arguments)
  return input_arguments


def add_training_episodes_parser(subparsers, agent_type_choices, environment_type_choices, reward_function_choices,
                                 grid_dimension_size_choices, episode_choices, max_timestep_choices, seed_choices):
  training_episodes_parser = subparsers.add_parser("training_episodes")

  training_episodes_parser.add_argument("agent_type", choices=agent_type_choices,
                                        help="type of agent to be used; should be a network agent if environment_type"
                                             " is not grid-world")
  training_episodes_parser.add_argument("environment_type", choices=environment_type_choices,
                                        help="type of environment to be used")
  training_episodes_parser.add_argument("reward_function", choices=reward_function_choices,
                                        help="type of reward function to be used; should be deterministic if"
                                             " environment_type is not grid-world")
  training_episodes_parser.add_argument("grid_dimension_size", type=int, choices=grid_dimension_size_choices,
                                        help="number of rows and columns in the square grid")
  training_episodes_parser.add_argument("episodes", type=int, choices=episode_choices,
                                        help="number of training episodes to simulate")
  training_episodes_parser.add_argument("max_timesteps", type=int, choices=max_timestep_choices,
                                        help="number of maximum timesteps in each training episode")
  training_episodes_parser.add_argument("seed", type=int, choices=seed_choices,
                                        help="seed value to control the randomness and get reproducible results")
  training_episodes_parser.add_argument("visual_evaluation_frequency", type=int,
                                        help="episode frequency for visually evaluating agent; should not be greater"
                                             " than value of episodes or negative; 0 suppresses visualization")
  training_episodes_parser.add_argument("-e", "--evaluate", action="store_true",
                                        help="evaluates trained agent")


def add_training_experiments_parser(subparsers, agent_type_choices, environment_type_choices, reward_function_choices,
                                    grid_dimension_size_choices, experiment_choices, timestep_choices):
  training_experiments_parser = subparsers.add_parser("training_experiments")

  training_experiments_parser.add_argument("agent_type", choices=agent_type_choices,
                                           help="type of agent to be used; should be a network agent if environment"
                                                " type is not grid-world")
  training_experiments_parser.add_argument("environment_type", choices=environment_type_choices,
                                           help="type of environment to be used")
  training_experiments_parser.add_argument("reward_function", choices=reward_function_choices,
                                           help="type of reward function to be used; should be deterministic if"
                                                " environment_type is not grid-world")
  training_experiments_parser.add_argument("grid_dimension_size", type=int, choices=grid_dimension_size_choices,
                                           help="number of rows and columns in the square grid")
  training_experiments_parser.add_argument("experiments", type=int, choices=experiment_choices,
                                           help="number of training experiments to simulate")
  training_experiments_parser.add_argument("timesteps", type=int, choices=timestep_choices,
                                           help="number of timesteps in each training experiment")
  training_experiments_parser.add_argument("output_path", help="path to directory in which to save output in; directory"
                                                               " will be created if it does not exist")
  training_experiments_parser.add_argument("-v", "--verbose", action="store_true",
                                           help="enables training information trace")


def add_training_timesteps_parser(subparsers, agent_type_choices, environment_type_choices, reward_function_choices,
                                  grid_dimension_size_choices, timestep_choices, seed_choices):
  training_timesteps_parser = subparsers.add_parser("training_timesteps")

  training_timesteps_parser.add_argument("agent_type", choices=agent_type_choices,
                                         help="type of agent to be used; should be a network agent if environment"
                                              " type is not grid-world")
  training_timesteps_parser.add_argument("environment_type", choices=environment_type_choices,
                                         help="type of environment to be used")
  training_timesteps_parser.add_argument("reward_function", choices=reward_function_choices,
                                         help="type of reward function to be used; should be deterministic if"
                                              " environment_type is not grid-world")
  training_timesteps_parser.add_argument("grid_dimension_size", type=int, choices=grid_dimension_size_choices,
                                         help="number of rows and columns in the square grid")
  training_timesteps_parser.add_argument("timesteps", type=int, choices=timestep_choices,
                                         help="number of training timesteps to simulate")
  training_timesteps_parser.add_argument("seed", type=int, choices=seed_choices,
                                         help="seed value to control the randomness and get reproducible results")
  training_timesteps_parser.add_argument("output_path", help="path to directory in which to save output in; directory"
                                                             " will be created if it does not exist")
  training_timesteps_parser.add_argument("-v", "--verbose", action="store_true",
                                         help="enables training information trace")
