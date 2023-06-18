# %%
from pathlib import Path

path_data = Path('./data/hand')

# %%
import numpy as np
X = np.array([
    np.loadtxt(filename, skiprows=1)
    for filename in sorted(path_data.glob("*.pts"))
])

# %%
original_shape = X.shape
original_shape

# %%
import matplotlib.pyplot as plt

for hand in X[:4]:
    plt.scatter(hand[:, 0], hand[:, 1])
plt.axis("square")

# %%
mean_hand = X.mean(axis=0)
plt.scatter(mean_hand[:, 0], mean_hand[:, 1])
plt.axis("square")

# %%
plt.scatter(mean_hand[:, 0], mean_hand[:, 1])
plt.scatter(hand[:, 0], hand[:, 1])
plt.axis("square")

# %%
from scipy.spatial import procrustes

xx, yy, zz = procrustes(mean_hand, X[0])

# %%
plt.scatter(xx[:, 0], xx[:, 1])
plt.scatter(yy[:, 0], yy[:, 1])
plt.axis("square")

# %%
X_registered = np.array([procrustes(mean_hand, hand)[1] for hand in X])

# %%
for hand in X_registered[:4]:
    plt.scatter(hand[:, 0], hand[:, 1])
plt.axis("square")

# %%
X_registered = X_registered.reshape(
    X_registered.shape[0], np.prod(X_registered.shape[1:])
)

# %%
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_registered)

# %%
mean_hand = pca.mean_.reshape(original_shape[1:])
plt.scatter(mean_hand[:, 0], mean_hand[:, 1])
plt.axis("square")

# %%
component = 0
component_hand = pca.components_[component, :].reshape(original_shape[1:])
l = 0.1

plt.plot(mean_hand[:, 0], mean_hand[:, 1])
plt.plot(mean_hand[:, 0] + l * component_hand[:, 0],
         mean_hand[:, 1] + l * component_hand[:, 1])
plt.plot(mean_hand[:, 0] - l * component_hand[:, 0],
         mean_hand[:, 1] - l * component_hand[:, 1])
plt.axis("square")

# %%
n_components = 5
fig, axs = plt.subplots(nrows=n_components, ncols=3,
                        sharex=True, sharey=True,
                        figsize=(7, 10))
for cmp, ax in zip(range(n_components), axs):
    component_hand = pca.components_[cmp, :].reshape(original_shape[1:])
    ax[0].plot(mean_hand[:, 0], mean_hand[:, 1])
    ax[1].plot(mean_hand[:, 0] + l * component_hand[:, 0],
               mean_hand[:, 1] + l * component_hand[:, 1])
    ax[2].plot(mean_hand[:, 0] - l * component_hand[:, 0],
               mean_hand[:, 1] - l * component_hand[:, 1])
    for x in ax:
        x.axis("square")

    if cmp == 0:
        ax[0].set_title("Mean hand")
        ax[1].set_title("Mean hand + lambda")
        ax[2].set_title("Mean hand - lambda")
    ax[0].set_ylabel(f"#{cmp + 1} component")
