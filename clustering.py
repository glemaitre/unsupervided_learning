# %% [markdown]
# # Clustering
#
# In this lecture, we will learn about clustering. First, we explain the general concept
# of clustering. Then, we implement the k-means algorithm from scratch. Finally, we use
# clustering in a couple of use cases.
#
# ## Clustering, an unsupervised learning task
#
# Clustering is an unsupervised learning task. We first explain the difference between
# supervised and unsupervised learning. Let's start by loading the iris dataset.

# %%
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
iris.frame

# %% [markdown]
#
# This dataset contains 150 samples of flowers. The flower features are the folllowing:

# %%
iris.feature_names

# %% [markdown]
#
# While the target corresponds to the flower species. To simplify the problem, we
# consider only two features. We select the petal length and width.

# %%
import pandas as pd

selected_features = ["petal length (cm)", "petal width (cm)"]
X = iris.data.loc[:, selected_features]
y = pd.Series(iris.target_names, name="target")[iris.target].reset_index(drop=True)

# %%
X

# %%
y

# %% [markdown]
#
# In a supervised learning task, both `X` and `y` are used to train the estimator.
# Visually, we have the following representation:

# %%
import seaborn as sns

sns.set_context("notebook")

_ = sns.scatterplot(
    data=pd.concat([X, y], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="target",
)

# %% [markdown]
#
# Associated with each 2-dimensional sample, we have a category corresponding to the
# flower species. In an unsupervised learning task, we do not have the target
# information, and thus the following information at hand:

# %%
_ = sns.scatterplot(data=X, x=selected_features[0], y=selected_features[1])

# %% [markdown]
# The goal of clustering is to find some strucutre in the data and group samples
# together.
#
# Having this representation can be useful notably for:
#
# - exploring data and finding patterns;
# - visualizations;
# - as preprocessing of a supervised learning task.
#
# Now, let's go into details in a simple but common clustering algorithm: k-means.
#
# ## K-means
#
# In this section, we guide you to implement an algorithm from scratch called k-means.
# The goal of k-means is to define k-centroids that will be iteratively updated such
# that the distance between the samples and the centroids is minimized.
#
# ### Our own implementation
#
# *Using numpy, draw three data samples from `X` that are used as initialization
# centroids*. Hint: you can use the function `np.random.choice` to draw samples.

# %%
import numpy as np

n_centroids = 3
init_centroids_idx = np.random.choice(X.index, size=n_centroids, replace=False)
init_centroids_idx

# %%
init_centroids = X.loc[init_centroids_idx]
init_centroids

# %% [markdown]
#
# *Plot the data samples and the centroids in a scatter plot*.

# %%
sns.scatterplot(data=X, x=selected_features[0], y=selected_features[1], label="data")
_ = sns.scatterplot(
    data=init_centroids,
    x=selected_features[0],
    y=selected_features[1],
    label="centroids",
)

# %% [markdown]
#
# Now, our job is to move those centroids such that the distance between the samples
# and the closest centroid is minimized.
#
# *Compute the distance between each sample in `X` and each centroid*. Hint: you can
# use the function `scipy.spatial.distance.cdist`.

# %%
import scipy as sp

dist_data_to_centroids = sp.spatial.distance.cdist(
    X, init_centroids, metric="euclidean"
)
dist_data_to_centroids.shape

# %%
dist_data_to_centroids

# %% [markdown]
#
# *Compute the averaged distance between each sample and the closest centroids.*
#
# In the next iteration, we are going to check that this distance is decreasing.

# %%
dist_data_to_centroids.min(axis=1).mean()

# %% [markdown]
#
# *Compute the label of the closest centroids for each data samples*. Hint: you can use
# the method `np.argmin`.

# %%
data_labeled = dist_data_to_centroids.argmin(axis=1)
data_labeled

# %% [markdown]
#
# *Plot the data samples with their associated labels, as well as the centroids in a
# scatter plot.*

# %%
sns.scatterplot(
    data=pd.concat([X, pd.Series(data_labeled, name="labels")], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="labels",
)
_ = sns.scatterplot(
    data=init_centroids,
    x=selected_features[0],
    y=selected_features[1],
    label="centroids",
)

# %% [markdown]
#
# Now, we go back to the start of algorithm and update the centroids to a better
# location. Indeed, we compute the mean location of the grouped samples.
#
# *Group the original data by labels and compute the mean sample for each group*. Hint:
# Be aware that you can leverage the original dataframe using `X.groupby(labels)` where
# `labels` corresponds to the labelled data from the previous step.

# %%
new_centroids = X.groupby(data_labeled).mean()
new_centroids

# %% [markdown]
#
# *Compute again the distance between each sample and the new centroids, the averaged
# distance between each sample and the closest centroid, and the label of the closest
# centroids for each data samples*. Hint: it corresponds to the three previous steps.

# %%
dist_data_to_centroids = sp.spatial.distance.cdist(X, new_centroids, metric="euclidean")
dist_data_to_centroids.min(axis=1).mean()

# %%
data_labeled = dist_data_to_centroids.argmin(axis=1)

# %% [markdown]
#
# *Plot the data samples with their associated labels, as well as the centroids in a
# scatter plot.*

# %%
sns.scatterplot(
    data=pd.concat([X, pd.Series(data_labeled, name="labels")], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="labels",
)
_ = sns.scatterplot(
    data=new_centroids,
    x=selected_features[0],
    y=selected_features[1],
    label="centroids",
)

# %% [markdown]
#
# *Repeat the previous steps by executing the cells multiple times until the centroids
# do not move anymore.* Is the error still decreasing?
#
# *Wrap the previous steps in a function called `k_means` that takes as arguments the
# data `X`, the number of clusters `n_clusters`, and the number of iterations to do. It
# should return the labeled data and the centroids.*
#
# *Plot the data samples with their associated labels, as well as the centroids in a
# scatter plot.*


# %%
def k_means(X, n_clusters, max_iter=100):
    centroids = X.loc[np.random.choice(X.index, size=n_clusters, replace=False)]
    for _ in range(max_iter):
        dist_data_to_centroids = sp.spatial.distance.cdist(
            X, centroids, metric="euclidean"
        )
        data_labeled = dist_data_to_centroids.argmin(axis=1)
        centroids = X.groupby(data_labeled).mean()

    return pd.Series(data_labeled, name="labels"), centroids


# %%
data_labeled, centroids = k_means(X, n_clusters=3)
sns.scatterplot(
    data=pd.concat([X, data_labeled], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="labels",
)
_ = sns.scatterplot(
    data=centroids,
    x=selected_features[0],
    y=selected_features[1],
    label="centroids",
)

# %% [markdown]
#
# *By running several time the algorithm above, does the labels of the data samples are
# always the same?*
#
# *Does the centroids are always the same?*
#
# Since we define to have three centroids (or clusters), then we can make a direct
# comparison with the original target.

# %%
_ = sns.scatterplot(
    data=pd.concat([X, y], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="target",
)

# %% [markdown]
#
# It is quite common to use a cluster metric that compare the results of the clustering
# with some actual target. The `adjusted_rand_score` is one of them. It will be equal to
# 1 if the clustering is identical to the target (up to a permutation).

# %%
from sklearn.metrics.cluster import adjusted_rand_score

adjusted_rand_score(y, data_labeled)

# %% [markdown]
#
# *Repeat the previous experiment using 5 clusters instead of 3 clusters.*

# %%
data_labeled, centroids = k_means(X, n_clusters=5)

sns.scatterplot(
    data=pd.concat([X, data_labeled], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="labels",
)
_ = sns.scatterplot(
    data=centroids,
    x=selected_features[0],
    y=selected_features[1],
    label="centroids",
)

# %%
adjusted_rand_score(y, data_labeled)

# %% [markdown]
#
# *Instead of using your own implementation, use the `KMeans` class from scikit-learn
# and check that you obtain the similar results.* Bonus question: what does the warning
# message mean?

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

data_labeled = kmeans.predict(X)

# %%
sns.scatterplot(
    data=pd.concat([X, pd.Series(data_labeled, name="labels")], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="labels",
)
_ = sns.scatterplot(
    data=pd.DataFrame(kmeans.cluster_centers_, columns=selected_features),
    x=selected_features[0],
    y=selected_features[1],
    label="centroids",
)

# %% [markdown]
#
# ### How to choose the number of clusters?
#
# From the previous experiment, we saw that we need to decide of the number of clusters
# to use. In practice, there is no perfect solution. A potential solution is to look at
# the inertia (the averaged dispersion of the data around each centroids) as a function
# of the number of clusters. Let's plot this curve.

# %%
n_clusters = range(1, 15)
intertia = [KMeans(n_clusters=k, n_init=1).fit(X).inertia_ for k in n_clusters]

# %%
import matplotlib.pyplot as plt

_, ax = plt.subplots()
ax.plot(n_clusters, intertia, marker="o")
ax.set_xlabel("n_clusters")
_ = ax.set_ylabel("inertia")

# %% [markdown]
#
# In this plot, we search for the "elbow" point, i.e. the point where the inertia does
# not decrease significantly anymore. In this case, we could choose 3 or 4 clusters.
#
# Sometimes, we compute some metrics and make a grid-search to find the optimal score.
#
# Some clustering algorithms do not require to specify the number of clusters. For
# instance, the DBSCAN algorithm will find the number of clusters based on the density
# of the data.

# %%
from sklearn.cluster import DBSCAN

dbscan = DBSCAN()
data_labeled = dbscan.fit_predict(X)

# %%
_ = sns.scatterplot(
    data=pd.concat([X, pd.Series(data_labeled, name="labels")], axis=1),
    x=selected_features[0],
    y=selected_features[1],
    hue="labels",
)

# %% [markdown]
#
# It is important to know that each clustering method comes with some assumptions
# regarding the definition of a cluster. We can have a look a the scikit-learn
# documentation:
# https://scikit-learn.org/dev/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
#
# ## Some applications of clustering
#
# Now, we are going to have a look to a couple of applications of clustering.
#
# ### Image compression
#
# Let's load an sample image available in scikit-learn.

# %%
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg")
china = china / china.max()
china.shape

# %% [markdown]
#
# We see that this is a 3-dimensional array. The first two dimensions correspond to the
# height and width of the image. The last dimension corresponds to the color channels
# (red, green, blue).

# %%
plt.imshow(china)
plt.axis("off")
print(f"Potential number of colors: {256 **3:,} colors")
print(
    f"Actual number of colors: {len(np.unique(china.reshape(-1, 3), axis=0)):,} colors"
)

# %% [markdown]
#
# We see that the image has a lot of colors. Indeed, we might not need so many colors to
# represent the image. We can use clustering to reduce the number of colors. It will
# group pixels of similar colors together. We will use the k-means algorithm to do so.
#
# *Using k-means, cluster the color of the image by using 64 colors.* Hint: remember
# that you need to reshape the image such that each pixel is a sample and the features
# are the color channels.

# %%
n_colors = 64
kmeans = KMeans(n_clusters=n_colors, n_init=1)
kmeans.fit(china.reshape(-1, 3))

# %% [markdown]
#
# *First get an image that contains the labels resulting from the clustering. Plot this
# image.* Hint: you can use the `predict` method of `KMeans` estimator that you fitted
# previously.

# %%
fake_image = kmeans.predict(china.reshape(-1, 3))
plt.imshow(fake_image.reshape(china.shape[:2]))
_ = plt.axis("off")

# %% [markdown]
# *Then, replcae each label by the centroid of the cluster. Plot the resulting image.*
# Hint: you can access the centroids using `kmeans.cluster_centers_`.

# %%
plt.imshow(kmeans.cluster_centers_[fake_image].reshape(china.shape))
_ = plt.axis("off")

# %% [markdown]
#
# ### Semi-supervised learning
#
# In this section, we see how to use clustering to perform semi-supervised learning.
# Here, we want to use a supervised preditive model but we do not have enough labeled
# data. We will use clustering to help us at having more data.
#
# We will use the digits dataset.

# %%
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

# %%
plt.imshow(X[0].reshape(8, 8), cmap="gray")
_ = plt.axis("off")

# %% [markdown]
#
# Let's first divide the data into a training and a testing set.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
#
# Now, let's imagine that we only have 50 labeled samples in our training set. We
# separate our training set accordingly.

# %%
n_observed = 50
X_observed = X_train[:n_observed]
X_unlabelled = X_train[n_observed:]
y_observed = y_train[:n_observed]
y_unlabelled = y_train[n_observed:]

# %% [markdown]
#
# *Create a predictive model made of `MinMaxScaler` and a `LogisticRegression`. Train
# it on the observed portion of the training set and evaluate its performance on the
# testing set.*

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(MinMaxScaler(), LogisticRegression())
model.fit(X_observed, y_observed)
model.score(X_test, y_test)

# %% [markdown]
#
# We see that the performance is quite low. We will now use clustering to help us.
# *Use `KMeans` with 50 clusters to cluster the entire training set.*

# %%
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, n_init=1).fit(X_train)

# %% [markdown]
#
# *Use `KMeans.transform` to get the distance of each sampe to each cluster. Then,
# find the closest sample to each cluster and select it as a prototype.*

# %%
prototype_idx = kmeans.transform(X_train).argmin(axis=0)
X_prototype, y_prototype = X_train[prototype_idx], y_train[prototype_idx]

# %% [markdown]
#
# *Use the previous prototype to train a new predictive model and evaluate its
# performance on the testing set. Does the performance improve?*

# %%
model.fit(X_prototype, y_prototype)
model.score(X_test, y_test)

# %% [markdown]
#
# ### Clustering as a preprocessing step
#
# A slightly different way of using clustering as a preprocessing step in the pipeline.
# Here, we compare two models, where one of the model will use `KMeans` and thus the
# distance of a sample to each cluster instead of the original features.

# %%
from sklearn.preprocessing import StandardScaler

model_with_clustering = make_pipeline(
    KMeans(n_clusters=n_clusters, n_init=1),
    StandardScaler(),
    LogisticRegression(max_iter=1_000),
)
model_without_clustering = make_pipeline(
    MinMaxScaler(), LogisticRegression(max_iter=1_000)
)

# %%
print(
    "Accuracy of the model with clustering as preprocessing: "
    f"{model_with_clustering.fit(X_train, y_train).score(X_test, y_test):.2f}"
)
print(
    "Accuracy of the model without clustering as preprocessing: "
    f"{model_without_clustering.fit(X_train, y_train).score(X_test, y_test):.2f}"
)


# %% [markdown]
#
# *Can we conclude that the model using clustering as a preprocessing step is better
# than the other one?*

# %%
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_results_with_clusterting = cross_validate(
    model_with_clustering, X_train, y_train, cv=cv, n_jobs=-1
)
cv_results_without_clusterting = cross_validate(
    model_without_clustering, X_train, y_train, cv=cv, n_jobs=-1
)

# %%
results = pd.DataFrame(
    {
        "With clustering": cv_results_with_clusterting["test_score"],
        "Without clustering": cv_results_without_clusterting["test_score"],
    }
).plot.box(whis=100)
_ = plt.title("Accuracy of the model with and without clustering as preprocessing")
