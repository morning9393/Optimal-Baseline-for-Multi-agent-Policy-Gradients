import torch
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic, R_Q_Critic
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        # self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.critic = R_Q_Critic(args, self.share_obs_space, self.act_space, self.device)
        self.ob_n_actions = args.ob_n_actions
        self.use_ob = args.use_ob
        self.actor_scheduler = None

        if self.act_space.__class__.__name__ == "Box":
            self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=self.lr, momentum=0.9)
            self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.critic_lr, momentum=0.9)
            self.actor_scheduler = lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=args.lr_decay)
            # self.critic_scheduler = lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                    lr=self.lr, eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self.critic_lr,
                                                     eps=self.opti_eps,
                                                     weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_continuous_ob(self, cent_obs, obs):
        ob_actions, _, _, mu, std = self.actor(obs, n_actions=self.ob_n_actions)
        ob_actions = ob_actions.permute(1, 0, 2)
        ob_actions_flatten = ob_actions.reshape(-1, self.act_space.shape[0])
        ob_cent_obs = np.expand_dims(cent_obs, 1).repeat(self.ob_n_actions, axis=1)
        ob_cent_obs_flatten = ob_cent_obs.reshape(-1, self.share_obs_space.shape[0])

        ob_qs_flatten, _ = self.critic(ob_cent_obs_flatten, act=ob_actions_flatten)
        ob_qs = ob_qs_flatten.reshape(-1, self.ob_n_actions, 1)

        ob_mu = mu.repeat(self.ob_n_actions, 1, 1).permute(1, 0, 2)
        ob_std = std.repeat(self.ob_n_actions, 1, 1).permute(1, 0, 2)

        ob = self.compute_continuous_ob(ob_qs, ob_actions, ob_mu, ob_std)
        return ob

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if self.act_space.__class__.__name__ == "Box":
            actions, action_log_probs, rnn_states_actor, mu, std = self.actor(obs,
                                                                              rnn_states_actor,
                                                                              masks,
                                                                              available_actions,
                                                                              deterministic)
            qs, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, act=actions)
            ob = qs  # here ob is dropped
            return qs, actions, action_log_probs, rnn_states_actor, rnn_states_critic, ob
        else:
            actions, action_log_probs, rnn_states_actor, pi = self.actor(obs,
                                                                         rnn_states_actor,
                                                                         masks,
                                                                         available_actions,
                                                                         deterministic)
            qs, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
            ob = qs
            if self.use_ob:
                ob = self.compute_discrete_ob(qs, pi)
                qs = qs.gather(-1, actions)
            return qs, actions, action_log_probs, rnn_states_actor, rnn_states_critic, ob

    def compute_continuous_ob(self, q, a, mu, std):
        normalize = lambda x: F.normalize(x, p=1, dim=-1)
        q = q.squeeze(-1)
        mu_term = torch.norm((a - mu) / std ** 2, dim=-1)
        std_term = torch.norm(((a - mu) ** 2 - std ** 2) / std ** 3, dim=-1)
        xweight = normalize(mu_term ** 2 + std_term ** 2)
        return (xweight * q).sum(-1).unsqueeze(-1).detach()

    def compute_discrete_ob(self, q, pi):
        """
        Calculating the optimal baseline
        :param q: Q values of all actions under the same states
        :param pi: Probability distributions of actions
        :return: The optimal baseline
        """
        M = torch.norm(pi, dim=-1) ** 2 + 1
        M = M.unsqueeze(-1)
        xweight = (M - 2 * pi) * pi
        enum = (xweight * q).sum(-1)
        deno = xweight.sum(-1)
        return (enum / deno).unsqueeze(-1).detach()

    def get_values(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                   available_actions=None, deterministic=False):
        """
        Get value function predictions.
        :param available_actions:
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.act_space.__class__.__name__ == "Box":
            actions, _, _, mu, std = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
            qs, _ = self.critic(cent_obs, rnn_states_critic, masks, act=actions)
            return qs
        else:
            actions, _, _, pi = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
            qs, _ = self.critic(cent_obs, rnn_states_critic, masks)
            if self.use_ob:
                qs = qs.gather(-1, actions)
            return qs

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        actions, action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)
        if self.act_space.__class__.__name__ == "Box":
            qs, _ = self.critic(cent_obs, rnn_states_critic, masks, act=action)
        else:
            qs, _ = self.critic(cent_obs, rnn_states_critic, masks)
            if self.use_ob:
                # qs = qs.gather(-1, torch.tensor(action, dtype=torch.int64))
                qs = qs.gather(-1, actions)
        return qs, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        if self.act_space.__class__.__name__ == "Box":
            actions, _, rnn_states_actor, _, _ = self.actor(obs, rnn_states_actor, masks, available_actions,
                                                            deterministic)
        else:
            actions, _, rnn_states_actor, _ = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
