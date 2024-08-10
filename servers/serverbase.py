import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from DDPG.DDPG import ExperienceReplayBuffer
from utils.data_utils import read_client_data, read_data
from utils.dlg import DLG
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import pickle
from datetime import datetime


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes

        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.num_sel_clients = args.num_sel_clients
        self.aggre_num = args.aggre_num

        self.algorithm = args.algorithm
        self.attack = args.attack
        self.goal = args.goal

        self.alpha = args.alpha
        self.seed = args.seed
        np.random.seed(args.seed)

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        self.selected_clients_idx = []
        self.ad_ratio = args.ad_ratio

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_std = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.all_reward = [[] for _ in range(self.global_rounds)]
        self.times = times
        self.replay_buffer = ExperienceReplayBuffer(args.aggre_num, args.aggre_num, max_size=args.buffer_size)

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               )
            self.clients.append(client)

    def get_all_clients_id(self):
        clients_id = []
        for c in self.clients:
            clients_id.append(c.id)
        return clients_id

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

    def receive_models_RL(self, action, aggre_clients):
        self.uploaded_ids = []
        self.uploaded_weights = action.reshape(-1)
        self.uploaded_models = []
        tot_samples = 0

        for client in aggre_clients:

            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        self.global_model_copy = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model_copy.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        for global_param, global_param_copy in zip(self.global_model.parameters(), self.global_model_copy.parameters()):
            global_param.data = (1 - self.alpha) * global_param_copy.data + self.alpha * global_param

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model_copy.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "./results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            time_now = datetime.now()
            algo = algo + "_" + str(time_now) + "_an:" + str(self.aggre_num) + '_ad:' + str(self.ad_ratio) + '_at:' + str(self.attack)

            file_path = result_path + "{}.h5".format(algo)
            reward_file_path = result_path + "{}.pkl".format(algo)
            print("File path: " + file_path)

            with open(reward_file_path, 'wb') as file:
                pickle.dump(self.all_reward, file)
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_std', data=self.rs_test_std)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def test_server(self, round_i):
        test_data_server = read_data(self.dataset, None, is_train=False)
        test_dataloader_server = DataLoader(test_data_server, batch_size=64, drop_last=False, shuffle=True)
        self.global_model.eval()

        test_acc = 0
        test_num = 0

        label_counts_dict = {str(i): 0 for i in range(self.num_classes)}
        with torch.no_grad():
            for x, y in test_dataloader_server:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.global_model(x)
                predictions = torch.argmax(output, dim=1)
                acc = predictions == y

                test_acc += (torch.sum(acc)).item()
                test_num += y.shape[0]
                rewards = y[acc]
                unique_labels, counts = torch.unique(rewards, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    label_counts_dict[str(label.item())] += count.item()

                label_counts_dict['reward'] = test_acc / test_num
                self.all_reward[round_i] = label_counts_dict

        return test_acc / test_num

    def test_metrics(self, benign_clients):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            if c.id in benign_clients:
                ct, ns, auc = c.test_metrics()
                tot_correct.append(ct * 1.0)
                tot_auc.append(auc * ns)
                num_samples.append(ns)

        ids = benign_clients

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self, benign_clients):
        num_samples = []
        losses = []
        for c in self.clients:
            if c.id in benign_clients:
                cl, ns = c.train_metrics()
                num_samples.append(ns)
                losses.append(cl * 1.0)

        ids = benign_clients

        return ids, num_samples, losses
    # evaluate selected clients

    def evaluate(self, benign_clients, acc=None, loss=None):
        stats = self.test_metrics(benign_clients)
        stats_train = self.train_metrics(benign_clients)

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        test_std = np.std([x / y if y != 0 else 0 for x, y in zip(stats[2], stats[1])])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
            self.rs_test_std.append(test_std)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        return stats, train_loss
