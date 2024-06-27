# -*- coding: utf-8 -*-
"""
Example of PWA regression to fit acrobot data.
Using PARC (C) 2021-2023 A. Bemporad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from pyparc.parc import PARC

np.random.seed(0)  # for reproducibility
savefigs = True # save figures as PNG images

nx = 2
nyc = 2 # number of numeric outputs

# Read dataset
file = open("acrobot_data.txt", "r") 
str = file.readlines()
X, Y = np.zeros(shape=(0, nx)), np.zeros(shape=(0, nyc))
for ln in str:
    ln = ln.replace("(", "")
    ln = ln.replace(")", "")
    ln = ln.replace(",", "")
    words = ln.split()
    assert len(words) == nx + nyc
    nums = [float(word) for word in words]
    x = [nums[0], nums[1]]
    y = [nums[2], nums[3]]
    X = np.vstack((X, np.array([x])))
    Y = np.vstack((Y, np.array([y])))

test_size = 0.2

xmin = min(X[:, 0])
xmax = max(X[:, 0])
ymin = min(X[:, 1])
ymax = max(X[:, 1])

# Plot level sets of nonlinear function
plt.close('all')
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.grid()

dx = (xmax - xmin) / 100.0
dy = (ymax - ymin) / 100.0
[x1, x2] = np.meshgrid(np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy))
nlevels = 8
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

if savefigs:
    plt.savefig('fig1.png')
else:
    plt.show()

# Setup PARC regression algorithm
K = 10          # maximum number of partitions
alpha = 1.0e-4  # L2-regularization
maxiter = 15    # maximum number of block-coordinate descent iterations

predictor = PARC(K=K, alpha=alpha, maxiter=maxiter)

# Solve PWA regression problem
categorical = False # we have a numeric target
predictor.fit(X, Y)

# Compute R2 scores
score_train = predictor.score(X, Y)

print("\nR2 scores:\n")
print("Training data: %6.2f %%" % (score_train[0] * 100))
print("--------------------\n")

Kf = predictor.K  # final number of partitions
delta = predictor.delta  # final assignment of training points to clusters
xbar = predictor.xbar  # centroids of final clusters

# Plot level sets of PWA prediction function
zpwl, _ = predictor.predict(np.hstack((x1.reshape(x1.size, 1), x2.reshape(x2.size, 1))))
zpwl1 = zpwl[:, 0].reshape(x1.shape)
zpwl2 = zpwl[:, 1].reshape(x1.shape)

fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.grid()
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
NN = x1.shape[0]
plt.contourf(x1, x2, zpwl1, alpha=0.6, levels=nlevels)
plt.contour(x1, x2, zpwl1, linewidths=3.0, levels=nlevels)
plt.title('PARC (K = %d)' % K, fontsize=20)

if savefigs:
    plt.savefig('fig2.png')
else:
    plt.show()

fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.grid()
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
NN = x1.shape[0]
plt.contourf(x1, x2, zpwl2, alpha=0.6, levels=nlevels)
plt.contour(x1, x2, zpwl2, linewidths=3.0, levels=nlevels)
plt.title('PARC (K = %d)' % K, fontsize=20)

if savefigs:
    plt.savefig('fig3.png')
else:
    plt.show()

Yh, _ = predictor.predict(X)

# Plot PWA partition
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(0, Kf):
    iD = (delta == i).ravel()
    plt.scatter(X[iD, 0], X[iD, 1], marker='*', linewidth=3,
                alpha=0.5, color=cm.tab10(i))
plt.grid()
plt.scatter(xbar[:, 0], xbar[:, 1], marker='o', linewidth=5, alpha=.5, color=(.8, .4, .4))

predictor.plot_partition([xmin, ymin], [xmax, ymax], fontsize=32,
                         alpha=.4, linestyle='-', linewidth=2.0,
                         edgecolor=(0, 0, 0), facecolor=(1,1,1))
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.title('PARC (K = %d)' % K, fontsize=20)

if savefigs:
    plt.savefig('fig4.png')
else:
    plt.show()

# Plot PWA function
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=20)
ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=20)
ax.set_zlabel(r'$y$', fontsize=20)
for i in range(0, Kf):
    iD = (delta == i).ravel()
    ax.scatter(X[iD, 0], X[iD, 1], Y[iD, 0], marker='*',
               linewidth=3, alpha=0.5, color=cm.tab10(i))

ax.plot_surface(x1, x2, zpwl1, alpha=0.5)
ax.view_init(35, -120)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.title('PARC (K = %d)' % K, fontsize=20)

if savefigs:
    plt.savefig('fig5.png')
else:
    plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=20)
ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=20)
ax.set_zlabel(r'$y$', fontsize=20)
for i in range(0, Kf):
    iD = (delta == i).ravel()
    ax.scatter(X[iD, 0], X[iD, 1], Y[iD, 1], marker='*',
               linewidth=3, alpha=0.5, color=cm.tab10(i))

ax.plot_surface(x1, x2, zpwl2, alpha=0.5)
ax.view_init(35, -120)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.title('PARC (K = %d)' % K, fontsize=20)

if savefigs:
    plt.savefig('fig6.png')
else:
    plt.show()