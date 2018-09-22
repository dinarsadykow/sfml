#!/usr/bin/python
# -*- coding: utf8-*-

from mpl_toolkits import mplot3d
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def sq_loss_demo(df_auto):

    beta0 = np.linspace(-8000, 6500, 100)
    beta1 = np.linspace(5000, 20000, 100)

    x = df_auto.loc[:, ['mileage']]
    x = (x-x.mean(axis=0))/x.std(axis=0)
    X = np.c_[x, np.ones(df_auto.shape[0])]


    B0, B1 = np.meshgrid(beta0, beta1)
    L = ((X.dot(np.r_[B0.reshape(1,-1), B1.reshape(1,-1)]) - df_auto.loc[:, 'price'].values.reshape(-1,1))**2).sum(axis=0)/(2*df_auto.shape[0])

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.view_init(40, 25)
    ax.plot_surface(B0, B1, L.reshape(B0.shape), alpha=0.3,)
    # ax.plot_(X, Y, Z)
    ax.set_xlabel(r'$\beta_0$')
    ax.set_ylabel(r'$\beta_1$')

    ax = fig.add_subplot(1, 2, 2)
    contour = ax.contour(B0, B1, L.reshape(B0.shape),)
    plt.clabel(contour, inline=1, fontsize=10)
    ax.set_xlabel(r'$\beta_0$')
    ax.set_ylabel(r'$\beta_1$')
    
    plt.show()

def grad_demo(df_auto, iters=1, alpha=0.001):
    
    beta0 = np.linspace(-8000, 6500, 100)
    beta1 = np.linspace(5000, 20000, 100)

    x = df_auto.loc[:, ['mileage']]
    x = (x-x.mean(axis=0))/x.std(axis=0)
    y = df_auto.loc[:, 'price'].values
    X = np.c_[x, np.ones(df_auto.shape[0])]


    B0, B1 = np.meshgrid(beta0, beta1)
    L = ((X.dot(np.r_[B0.reshape(1,-1), B1.reshape(1,-1)]) - y.reshape(-1,1))**2).sum(axis=0)/(2*df_auto.shape[0])

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X[:,0], y)
    
    Beta, costs, Betas = gradient_descent_upd(X, y, alpha, tol=10**-3, max_iter=iters)
    Betas = np.c_[Betas]
    
    X_1 = np.sort(X, axis=0)
    
    y_hat = X_1.dot(Betas.T)
    
    plt.plot(X_1[:,0], y_hat)
    ax = fig.add_subplot(1, 2, 2)
    contour = ax.contour(B0, B1, L.reshape(B0.shape),)
    plt.clabel(contour, inline=1, fontsize=10)
    ax.set_xlabel(r'$\beta_0$')
    ax.set_ylabel(r'$\beta_1$')
    
    ax.plot(Betas[:,0], Betas[:, 1], '*-')
    
    plt.show()
    
    
def stoch_grad_demo(df_auto, iters=1, alpha=0.001):
    
    beta0 = np.linspace(-8000, 5500, 100)
    beta1 = np.linspace(5000, 20000, 100)

    x = df_auto.loc[:, ['mileage']]
    x = (x-x.mean(axis=0))/x.std(axis=0)
    y = df_auto.loc[:, 'price'].values
    X = np.c_[x, np.ones(df_auto.shape[0])]


    B0, B1 = np.meshgrid(beta0, beta1)
    L = ((X.dot(np.r_[B0.reshape(1,-1), B1.reshape(1,-1)]) - y.reshape(-1,1))**2).sum(axis=0)/(2*df_auto.shape[0])

    fig = plt.figure(figsize=(14, 7))    
    Beta, costs, Betas = gradient_descent_upd(X, y, alpha, tol=10**-3, max_iter=iters)
    Betas = np.c_[Betas]
    
    _, _, Betas_stoch = stoch_gradient_descent(X, y, alpha, max_iter=iters)
    Betas_stoch = np.c_[Betas_stoch]
    
    X_1 = np.sort(X, axis=0)
    
    y_hat = X_1.dot(Betas.T)
    ax = fig.add_subplot(1, 1, 1)
    contour = ax.contour(B0, B1, L.reshape(B0.shape),)
    plt.clabel(contour, inline=1, fontsize=10)
    ax.set_xlabel(r'$\beta_0$')
    ax.set_ylabel(r'$\beta_1$')
    
    ax.plot(Betas[:,0], Betas[:, 1], '*-')
    ax.plot(Betas_stoch[:,0], Betas_stoch[:, 1], 'o-', c='b')
    plt.axis('equal')
    
    plt.show()
    

def gradient_descent_upd(X, y, alpha, tol=10**-3, max_iter=10):
    n = y.shape[0]
    Beta = np.array([-4000, 6000])
    delta = 10
    cost_prev = 0
    i = 0
    
    # for logging
    Betas = [Beta]
    costs = []
    
    while (delta > tol) and (i <= max_iter):
        y_hat = X.dot(Beta)
        
        # считаем ошибку и значение функции потерь
        error = (y_hat - y)
        cost = np.sum(error ** 2)/float(2 * n)
        delta = abs(cost - cost_prev)
        cost_prev = cost
        
        # считаем градиент
        grad = X.T.dot(error) / n

        # обновляем коэффициенты
        Beta = Beta - alpha * grad
        
        # logging
        if i % 5 == 0:
            costs.append(cost)
            Betas.append(Beta)
        i += 1
        
    return Beta, costs, Betas


def stoch_gradient_descent(X, y, alpha, max_iter=10):
    n = y.shape[0] 
    Beta = np.array([-4000, 6000])
    
    costs = []
    Betas = [Beta]
    
    for i in xrange(max_iter):
        
        X, y = shuffle(X, y, random_state=10)
        
        for j in range(n):
            
            y_hat = X[j].dot(Beta)

            # считаем ошибку и значение функции потерь
            error = y_hat - y[j]

            # считаем градиент
            gradient = X[j].T.dot(error)

            # обновляем коэффициенты
            Beta = Beta - alpha * gradient  # update
            alpha *= 0.99
                # logging
            if j % 5 == 0 and i % 5 == 0:
                Betas.append(Beta)
        
        cost_epoch = np.sum((X.dot(Beta) - y)**2 / (2*n))
        costs.append(cost_epoch)
        
        
    return Beta, costs, Betas