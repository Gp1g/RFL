import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math

NUM_USER = 100
dir_path = "./synthetic11/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Setup directory for train/test data
config_path = dir_path + "config.json"
train_path = dir_path + "train/"
test_path = dir_path + "test/"
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(alpha, beta, iid):
    dimension = 60
    NUM_CLASS = 10

    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    print(samples_per_user)
    min_ = samples_per_user.min()
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        mean_x[i] = np.random.normal(B[i], 1, dimension)
        print(mean_x[i])

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1, NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))

    return X_split, y_split, min_


def main():
    X, y, min_ = generate_synthetic(alpha=1, beta=1, iid=0)  # synthetic (1,1)
    min_ = int(0.1 * min_)
    # Create data structure
    x_server = []
    y_server = []

    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:03d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        server_len = min_
        num_samples = num_samples - min_
        train_len = int(0.9 * num_samples)

        x_server.extend(np.array(X[i][:server_len]))
        y_server.extend(np.array(y[i][:server_len]))

        train_data = {'x': np.array(X[i][server_len:train_len + server_len]),
                      'y': np.array(y[i][server_len:train_len + server_len])}
        test_data = {'x': np.array(X[i][train_len + server_len:]), 'y': np.array(y[i][train_len + server_len:])}

        with open(train_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_data)
        with open(test_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_data)

    server_test = {'x': x_server, 'y': y_server}
    np.savez_compressed(test_path + 'server_test.npz', **server_test)


if __name__ == "__main__":
    main()