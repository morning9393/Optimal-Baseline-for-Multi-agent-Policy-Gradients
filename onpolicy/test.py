from onpolicy.envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import math
import gym

# env = gym.make("Walker2d-v2")
# env.seed(10)
# np.random.seed(10)
# print("act space: ", env.action_space)
# env.reset()
# action = np.random.uniform(-1.0, 1.0, 6)
# print("action: ", action)
# obs, r, d, info = env.step(action)
# print('r: ', r)
# print("d:", d)
# print("env health: ", info)


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


mu = 0
sigma = 0.35

print("log: ", np.log(0.56))

x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
y_sig = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.vlines(mu, 0, np.exp(-(mu - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma), colors="c",
           linestyles="dashed")
plt.vlines(mu + sigma, 0, np.exp(-(mu + sigma - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma),
           colors="k", linestyles="dotted")
plt.vlines(mu - sigma, 0, np.exp(-(mu - sigma - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma),
           colors="k", linestyles="dotted")
# plt.xticks([mu - sigma, mu, mu + sigma], ['μ-σ', 'μ', 'μ+σ'])
plt.xlabel('Frequecy')
plt.ylabel('Latent Trait')
plt.title('Normal Distribution: $\mu = %.2f, $sigma=%.2f' % (mu, sigma))
plt.grid(True)
plt.show()



# def main():
#     env_args = {"scenario": "Walker2d-v2",
#                   "agent_conf": "2x3",
#                   "agent_obsk": 0,
#                   "episode_limit": 1000}
#     env = MujocoMulti(env_args=env_args)
#     env_info = env.get_env_info()
#     env.seed(10)
#
#     n_actions = env_info["n_actions"]
#     print("n_actions: ", n_actions)
#     n_agents = env_info["n_agents"]
#     print("n_agents: ", n_agents)
#
#     print("obs sp: ", env.observation_space)
#     print("act sp: ", env.action_space)
#     n_episodes = 1
#
#     for e in range(n_episodes):
#         env.reset()
#         terminated = False
#         step = 0
#         while not terminated:
#             actions = []
#             for agent_id in range(n_agents):
#                 # action = np.random.uniform(-1.0, 1.0, n_actions)
#                 action = np.ones(n_actions) * 20
#                 actions.append(action)
#             print("actions: ", actions)
#             _, _, reward, terminated, _, _ = env.step(actions)
#             terminated = np.all(terminated)
#             print("reward: ", reward)
#             break
#             # env.render()
#             if terminated:
#                 print("total step: ", step)
#             else:
#                 step += 1
#             time.sleep(0.1)
#
#     env.close()
#
# if __name__ == "__main__":
#     main()

# toy_q = np.random.uniform(low=-1, high=1, size=(2, 10))
# toy_pi = np.random.random(size=(2, 10))
# toy_pi = toy_pi / sum(toy_pi)
# toy_q = torch.Tensor(toy_q)
# toy_pi = torch.Tensor(toy_pi)
#
# print("toy q: ", toy_q)
# print("toy pi: ", toy_pi)
#
# normalize = lambda x: F.normalize(x, p=1, dim=-1)
#
#
# def optimal_baseline(q, pi):
#     M = torch.norm(pi, dim=-1, keepdim=True) ** 2 + 1
#     xweight = normalize((M - 2 * pi) * pi)
#     return (xweight * q).sum(-1)
#
#
# def compute_ob(q, pi):
#     """
#     Calculating the optimal baseline
#     :param q: Q values of all actions under the same states
#     :param pi: Probability distributions of actions
#     :return: The optimal baseline
#     """
#     M = torch.norm(pi, dim=-1) ** 2 + 1
#     M = M.unsqueeze(-1)
#     xweight = (M - 2 * pi) * pi
#     enum = (xweight * q).sum(-1)
#     deno = xweight.sum(-1)
#     return enum / deno
#
#
# print("discrete ob in overleaf: ", optimal_baseline(toy_q, toy_pi))
# print("discrete ob implementation: ", compute_ob(toy_q, toy_pi))

# each action has four dimensions
# np.random.seed(0)
# toy_mu = np.random.uniform(low=-1, high=1, size=4)
# toy_std = np.random.uniform(low=0.5, high=2, size=4)
#
# toy_q = np.random.uniform(low=-1, high=1, size=10)
# toy_a = np.array([np.random.normal(toy_mu[i], toy_std[i], size=10) for i in range(4)]).T
# # toy_a = np.random.normal(toy_mu, toy_std, size=10)
# toy_a = toy_a / sum(toy_a)
#
# toy_q = torch.Tensor(toy_q)
# toy_a = torch.Tensor(toy_a)
# toy_mu = torch.Tensor(toy_mu)
# toy_std = torch.Tensor(toy_std)
#
# print("toy mu: ", toy_mu)
# print("toy std: ", toy_std)
# print("toy q: ", toy_q)
# print("toy a: ", toy_a)
#
#
# normalize = lambda x: F.normalize(x, p=1, dim=-1)
#
#
# def optimal_baseline(a, q, mu, std):
#     mu_term = torch.norm((a - mu) / std ** 2, p=1, dim=-1)
#     std_term = torch.norm(((a - mu) ** 2 - std ** 2) / std ** 3, p=1, dim=-1)
#     print("norm1: ", mu_term ** 2 + std_term ** 2)
#     xweight = normalize(mu_term ** 2 + std_term ** 2)
#     return (xweight * q).sum(-1)
#
#
# def get_ob(a, q, mu, std):
#     deno = 3 * torch.norm(1 / std, p=1) ** 2
#     sample_norm = torch.norm((a - mu) / std ** 2, p=1, dim=-1) ** 2 + torch.norm(
#         ((a - mu) ** 2 - std ** 2) / (std ** 3), p=1, dim=-1) ** 2
#     print("norm2: ", sample_norm)
#     enu = (q * sample_norm).mean()
#     ob = enu / deno
#     return ob
#
#
# print("continuous ob in overleaf: ", optimal_baseline(toy_a, toy_q, toy_mu, toy_std))
# print("continuous ob implementation: ", get_ob(toy_a, toy_q, toy_mu, toy_std))
