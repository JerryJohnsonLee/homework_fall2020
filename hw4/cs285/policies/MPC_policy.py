import numpy as np
from numpy.core.fromnumeric import size
from torch import random

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high]
        random_action_sequences = np.random.uniform(low=self.low, high=self.high, size=(num_sequences, horizon, self.ac_dim))
        return random_action_sequences

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon)

        # for each model in ensemble:
        predicted_sum_of_rewards_per_model = []
        for model in self.dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(
                obs, candidate_action_sequences, model)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)

        # calculate mean_across_ensembles(predicted rewards)
        predicted_rewards = np.mean(
            predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N

        # pick the action sequence and return the 1st element of that sequence
        best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)]   # TODO (Q2)
        action_to_take = best_action_sequence[0]  # TODO (Q2)
        return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        rewards = []
        obs = np.array([obs] * candidate_action_sequences.shape[0])
        not_done_indices = np.array([True] * candidate_action_sequences.shape[0])
        for h in range(candidate_action_sequences.shape[1]):
            current_actions = candidate_action_sequences[not_done_indices, h]
            timestep_rewards, done = self.env.get_reward(obs, current_actions)
            timestep_all_rewards = np.zeros_like(not_done_indices, dtype=timestep_rewards.dtype) - 5 # Make sure ending an episode early is bad
            timestep_all_rewards[not_done_indices] = timestep_rewards
            rewards.append(timestep_all_rewards)
            not_done_actions = current_actions[np.logical_not(done.astype(bool))]
            not_done_obs = obs[np.logical_not(done.astype(bool))]
            obs = model.get_prediction(not_done_obs, not_done_actions, self.data_statistics)
            not_done_indices[not_done_indices] = np.logical_not(done.astype(bool))
        sum_of_rewards = np.sum(rewards, axis=0)  # TODO (Q2)
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        return sum_of_rewards
