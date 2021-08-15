from .distributions import Categorical, DiagGaussian
import torch.nn as nn


class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()
        self.continuous = False
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)
        elif action_space.__class__.__name__ == "Box":
            self.continuous = True
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)
        else:
            raise NotImplementedError
    
    def forward(self, x, available_actions=None, deterministic=False, n_actions=0):
        """
        Compute actions and action logprobs from given input.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        action_logits = self.action_out(x, available_actions)
        if deterministic:
            actions = action_logits.mode()
        elif n_actions > 0:
            actions = action_logits.sample(sample_shape=(n_actions,))
        else:
            actions = action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)

        if self.continuous:
            return actions, action_log_probs, action_logits.mean, action_logits.stddev
        else:
            return actions, action_log_probs, action_logits.probs

    def get_probs(self, x, available_actions=None):
        """
        Compute action probabilities from inputs.
        :param x: (torch.Tensor) input to network.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)

        :return action_probs: (torch.Tensor)
        """
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs
        
        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, deterministic=False):
        """
        Compute log probability and entropy of given actions.
        :param x: (torch.Tensor) input to network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(action)

        if active_masks is not None:
            if self.continuous:
                dist_entropy = (action_logits.entropy()*active_masks).sum()/active_masks.sum()
            else:
                dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()
        
        return actions, action_log_probs, dist_entropy
