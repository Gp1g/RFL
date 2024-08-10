#!/usr/bin/env python
import copy
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from servers.serverRFL import RFL

from utils.models import *
from utils.resnet import *


from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

def run(args):
    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if "mnist" == args.dataset:
            args.model = OneHiddenLayerFc(784, 10).to(args.device)
        elif "emnist" == args.dataset:
            args.model = TwoHiddenLayerFc(784, 62).to(args.device)
        elif "Cifar10" == args.dataset:
            args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        elif "synthetic" in args.dataset:
            args.model = Logistic(60, out_dim=args.num_classes).to(args.device)
        elif "fashionmnist" == args.dataset:
            args.model = ResNet2(1, args.num_classes).to(args.device)

        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "RFL":
            server = RFL(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)

    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")
    parser.add_argument("--actor_lr", default=1e-2, type=float,
                        help='Learning rate for the networks (default: 0.001)')
    parser.add_argument("--critic_lr", default=1e-2, type=float,
                        help='Learning rate for the networks (default: 0.001)')
    parser.add_argument("--actor_decay", default=1e-5, type=float,
                        help='Decay rate for the networks (default: 0.00001)')
    parser.add_argument("--critic_decay", default=1e-5, type=float,
                        help='Decay rate for the networks (default: 0.00001)')
    parser.add_argument("--episodes", default=0, type=float,
                        help='Total episodes of the agent (default: 0)')
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-seed', "--seed", type=int, default=1)
    parser.add_argument("--attack", type=str, default='same_value',
                        help='type of the attack', choices=['same_value', 'sign_flip', 'gaussian'])
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_epochs", type=int, default=20,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="RFL")
    parser.add_argument('-nc', "--num_clients", type=int, default=100,
                        help="Total number of clients")
    parser.add_argument('-ns', "--num_sel_clients", type=int, default=100,
                        help="The number of selected clients")
    parser.add_argument('-an', "--aggre_num", type=int, default=50,
                        help="The number of clients aggregated")
    parser.add_argument("--ad_ratio", type=float, default=0.2,
                        help='Ratio of adversaries')
    parser.add_argument("--buffer_size", default=100000, type=int,
                        help='Size of the experience replay buffer (default: 100000)')
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")

    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')

    parser.add_argument('-al', "--alpha", type=float, default=0.)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("The number of aggregation clients: {}".format(args.aggre_num))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))

    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    print("=" * 50)

    run(args)

