# %% [markdown]
#
# ## Dimensionality reduction
#
# In this lecture, we briefly present the concept of dimensionality reduction.
# Dimensionality reduction serves three main purposes:
#
# * It can be used to visualize high dimensional data into 2D or 3D in order to
#   get some insights about the data.
# * It can be used to remove some noise from the data. In this case, the
#   dimensionality reduction algorithm will be used as a pre-processing step
#   before training a supervised learning model.
# * It can alleviate the curse of dimensionality. In this case, the
#   dimensionality reduction algorithm will be used as a pre-processing step
#   before training a supervised learning model.
#
# We will present a first technique called Principal Component Analysis (PCA).
#
# ## Principal Component Analysis
#
# The idea behind PCA is to find a new set of axes, ordered by importance, on
# which we can project the data. The first axis will be the axis along which
# the data vary the most. The second axis will be the axis orthogonal to the
# first one that explain the largest amount of remaining variance. And so on
# until we have as many axes as original features.
#
# Let's generate a ellipsoid dataset not centered at (0, 0) to illustrate the
# concept of PCA.

# %%
import numpy as np

rng = np.random.default_rng(0)
mean = [10, 10]
cov = np.array([[6, -3], [-3, 3.5]])
X = rng.multivariate_normal(mean=mean, cov=cov, size=1_000)

# %%
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.axis("square")

# %% [markdown]
#
# So here now, we will apply PCA on this dataset. We will keep the same number
# of dimensions as the original dataset and just observe what type of projection
# we obtain.

# %%
from sklearn.decomposition import PCA

pca = PCA()
X_transformed = pca.fit_transform(X)

# %%
_, axs = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

axs[0].scatter(X[:, 0], X[:, 1])
axs[0].set_xlabel("Original feature 1")
axs[0].set_ylabel("Original feature 2")

axs[1].scatter(X_transformed[:, 0], X_transformed[:, 1])
axs[1].set_xlabel("First principal component")
_ = axs[1].set_ylabel("Second principal component")

# %% [markdown]
#
# So we see that PCA will apply a rotation and a scaling to the original data
# such that the first axis will be the axis along which the data vary the most.
# The second axis will be the axis orthogonal to the first one that explain the
# largest amount of remaining variance. And so on until we have as many axes as
# original features.
#
# Indeed, we can get this information by inspecting the `explained_variance_`
# attribute of the PCA object.

# %%
pca.explained_variance_

# %%
pca.explained_variance_ratio_

# %% [markdown]
#
# So the first component explains more than 85% of the variance in the data.
#
# *Load the iris dataset and apply a PCA to project the 4-dimensional data to
# a 2-dimensional space. Plot the projected data.*

# %%
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# %%
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# %%
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)

# %%
import seaborn as sns

sns.pairplot(data=iris.frame, hue="target")
