import time
import random
import torch
import numpy as np
from clients.clientRFL import clientRFL
from servers.serverbase import Server
from threading import Thread
from DDPG import DDPG, environment
import torch.nn.functional as F


class RFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientRFL)  # create clients
        self.agent = DDPG.DDPG(args)
        self.env = environment.FedEnv(args, times)
        self.episodes = args.episodes
        self.batch_size = args.batch_size
        print("Finished creating server and clients.")

        self.Budget = []
        self.aggre_clients = [[] for _ in range(args.aggre_num)]
        self.all_clients_id = self.get_all_clients_id()
        self.ad_clients = [self.all_clients_id[i] for i in np.random.choice(len(self.all_clients_id), round(
            len(self.all_clients_id) * self.ad_ratio), replace=False)]
        self.benign_clients = list(filter(lambda i: i not in self.ad_clients, self.all_clients_id))
        self.aggre_num = args.aggre_num

    def get_state(self):
        random_indices = torch.tensor(random.sample(range(0, self.num_clients), self.num_sel_clients))
        self.selected_clients = [self.clients[i] for i in random_indices]
        all_client_param = [[] for _ in range(len(self.selected_clients))]

        for num, client in enumerate(self.selected_clients):
            for param in client.model.parameters():
                if param.data is not None:
                    all_client_param[num].append(param.data.detach().view(-1))

        all_client_param_temp = torch.stack([torch.cat(inner_list) for inner_list in all_client_param])

        state_temp = torch.zeros(len(self.selected_clients), len(self.selected_clients))
        for i in range(len(all_client_param_temp)):
            for j in range(len(all_client_param_temp)):
                state_temp[i][j] = torch.norm(all_client_param_temp[i] - all_client_param_temp[j])
        state_temp_sum = torch.sum(state_temp, dim=1)

        _, indices = torch.topk(state_temp_sum, k=self.aggre_num, largest=False)
        state = state_temp_sum[indices]
        state /= state.max()
        self.aggre_clients = [self.selected_clients[i] for i in indices]

        return state, self.aggre_clients

    def train(self):
        for episode in range(self.episodes+1):
            print(f"\n-------------Episode number: {episode}-------------")
            self.env.reset(self.global_model)
            done = 0
            self.send_models()
            state = torch.full((1, len(self.aggre_clients)), 1.).view(-1)
            random_indices = torch.tensor(random.sample(range(0, self.num_clients), self.aggre_num))
            self.aggre_clients = [self.clients[i] for i in random_indices]
            for i in range(self.global_rounds):
                s_t = time.time()
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")

                action = self.agent.select_action(state)
                self.receive_models_RL(action, self.aggre_clients)
                self.aggregate_parameters()
                self.send_models()
                self.evaluate(self.benign_clients)
                reward, opt_reward = self.test_server(i), 1
                print('Reward {:.4f}'.format(reward))

                if reward == opt_reward:
                    done = 1
                train_time_begin = time.time()
                for client in self.clients:
                    client.train()
                    if client.id in self.ad_clients:
                        if self.attack == 'same_value':
                            new_value = 1 * np.random.normal(0, 100, 1).item()
                            for param in client.model.parameters():
                                param.data.fill_(new_value)

                        elif self.attack == 'sign_flip':
                            magnitude = abs(np.random.normal(0, 10, 1).item())
                            for param in client.model.parameters():
                                param.data.mul_(magnitude)

                        elif self.attack == 'gaussian':
                            mean = 0
                            std = 100.
                            for param in client.model.parameters():
                                param.data = torch.randn(param.data.size()) * std + mean
                                client.model.to(self.device)

                train_time = time.time()
                print('train time {:.4f}'.format(train_time - train_time_begin))
                print("\nEvaluate local model")
                self.evaluate(self.benign_clients)

                next_state, self.aggre_clients = self.get_state()

                self.replay_buffer.add(state, action, next_state, reward, done)
                self.agent.update_parameters(self.replay_buffer, 16)

                state = next_state

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
                if done:
                    break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()



