import torch
from servers.serverbase import Server
import torch.nn as nn


class FedEnv(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.action_dim = args.num_clients
        self.state_dim = args.num_sel_clients

        self.state = None
        self.done = None

        self.episode_t = None

    def reset(self, model):
        self.episode_t = 0

        model.apply(weights_init)

    def compute_reward(self, stats):
        opt_reward = sum(stats[1])*1.0
        reward = sum(stats[2])*1.0

        return reward, opt_reward

    def step(self, stats_train):
        self.episode_t += 1

        reward, opt_reward = self.compute_reward(stats_train)

        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)