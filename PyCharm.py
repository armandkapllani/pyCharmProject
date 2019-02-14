#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:01:28 2018

@author: armandkapllani
algorithm: gradient descend (simple linear regression)
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats
from sklearn.datasets.samples_generator import make_regression

x, y = make_regression(n_samples=100,
                       n_features=1,
                       n_informative=1,
                       noise=20,
                       random_state=2017)


# ----------------------------#
# Gradient Descent Algorithm #
# ----------------------------#

# A simple linear regression model. 


def gradient_descent(x, y, theta_init, step=0.001, maxsteps=0, precision=0.001, ):
    costs = []
    m = y.size  # number of data points
    theta = theta_init
    history = []  # to store all thetas
    preds = []
    counter = 0
    oldcost = 0
    pred = np.dot(x, theta)
    error = pred - y
    currentcost = np.sum(error ** 2) / (2 * m)
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter += 1
    while abs(currentcost - oldcost) > precision:
        oldcost = currentcost
        gradient = x.T.dot(error) / m
        theta = theta - step * gradient  # update
        history.append(theta)

        pred = np.dot(x, theta)
        error = pred - y
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)

        if counter % 25 == 0: preds.append(pred)
        counter += 1
        if maxsteps:
            if counter == maxsteps:
                break

    return history, costs, preds, counter


xaug = np.c_[np.ones(x.shape[0]), x]
theta_i = [-15, 40] + np.random.rand(2)
history, cost, preds, iters = gradient_descent(xaug, y, theta_i, step=0.1)
theta = history[-1]
theta   



