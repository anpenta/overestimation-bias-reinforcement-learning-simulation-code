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

# Simulate Module
# Module to run reinforcement learning simulations.

import simulations
import utility

input_arguments = utility.parse_input_arguments()

state_encoding = "array" if "network" in input_arguments.agent_type else "integer"
environment = utility.create_environment(input_arguments.environment_type, input_arguments.grid_dimension_size,
                                         input_arguments.reward_function, state_encoding)

if input_arguments.simulation_function == "training_episodes":
  # Control the randomness to get reproducible results.
  utility.control_randomness(input_arguments.seed)

  # Create the agent and run the training episodes.
  agent = utility.create_agent(input_arguments.agent_type, environment.compute_state_space_size(),
                               environment.compute_action_space_size())
  simulations.simulate_training_episodes(agent, environment, episodes=input_arguments.episodes,
                                         max_timesteps=input_arguments.max_timesteps,
                                         visual_evaluation_frequency=input_arguments.visual_evaluation_frequency,
                                         evaluate_trained_agent=input_arguments.evaluate)

elif input_arguments.simulation_function == "training_experiments":
  # Run the training experiments.
  simulations.simulate_training_experiments(input_arguments.agent_type, environment,
                                            experiments=input_arguments.experiments,
                                            timesteps=input_arguments.timesteps, verbose=input_arguments.verbose,
                                            output_path=input_arguments.output_path)

elif input_arguments.simulation_function == "training_timesteps":
  # Control the randomness to get reproducible results.
  utility.control_randomness(input_arguments.seed)

  # Create the agent and run the training timesteps.
  agent = utility.create_agent(input_arguments.agent_type, environment.compute_state_space_size(),
                               environment.compute_action_space_size())
  simulations.simulate_training_timesteps(agent, environment, timesteps=input_arguments.timesteps,
                                          verbose=input_arguments.verbose, output_path=input_arguments.output_path)
