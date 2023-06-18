# %% [markdown]
#
# # Anomaly Detection
#
# In this notebook, we present the principle of anomaly detection. First, we will
# make the distinction between outlier detection and novelty detection.
#
# ## Outlier Detection
#
# Outlier detection is the task of identifying samples that are rare compared to the
# majority of the data. The samples that are rare could be called "outliers" or
# "anomalies".
#
# It is therefore important to understand that in this context, we have a contaminated
# training set on which we want to detect outliers. Therefore, the outlier detection
# algorithm can be applied in two contexts: as a preprocessing step of a supervised
# to clean the training data or as an unsupervised learning task for which we do not
# have labels but we know that the dataset is contaminated.
#
# Let's look at an example of credit card fraud:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# %%
import pandas as pd

df = pd.read_csv("data/creditcard.csv", index_col=0)
df.head()

# %%
X, y = df.drop(columns="Class"), df["Class"]

# %% [markdown]
#
# We use an `IsolationForest` to detect outliers. This algorithm is based on
# randomized trees. It create random split at each node of a tree. It has the effect
# of isolating outliers early in the tree.

# %%
from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(n_estimators=100, n_jobs=-1, random_state=0)
y_pred = outlier_detector.fit_predict(X)

# %%
y.value_counts()

# %% [markdown]
#
# The forest output 1 when we have an inlier and otherwise -1. We will switch to a 0/1
# encoding to be able to compute some standard classification scores.

# %%
from collections import Counter

Counter(y_pred)

# %%
import numpy as np

y_pred = np.where(y_pred == 1, 0, 1)
Counter(y_pred)

# %%
from sklearn.metrics import classification_report

print(classification_report(y, y_pred))

# %% [markdown]
# This strategy is not very effective here. Indeed, since we have the label information,
# it would be much better to use some supervised learning approach. Here are two
# potential examples:
#
# * https://imbalanced-learn.org/dev/auto_examples/applications/plot_outlier_rejections.html#sphx-glr-auto-examples-applications-plot-outlier-rejections-py
# * https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_anomaly_comparison.html#sphx-glr-auto-examples-miscellaneous-plot-anomaly-comparison-py
#
# ## Novelty Detection
#
# The novelty detection context is slightly different from the outlier detection
# context. Indeed, we expect the training data not to be contaminated with some outliers.
# However, on some new set of data, we expect to have some newly and differently
# distributed data and we would like to detect them.
#
# In this context, we expect to call `fit` on the training data and apply `predict` on
# some new dataset. One potential example could be to detect if the data coming in
# production is different from the data used during the development phase.

# %%
from sklearn.datasets import make_moons, make_blobs

moons, _ = make_moons(n_samples=500, noise=0.05)
blobs, _ = make_blobs(
    n_samples=500, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25
)
X_train = np.vstack([moons, blobs])
y_train = np.hstack(
    [
        np.ones(moons.shape[0], dtype=np.int8),
        np.zeros(blobs.shape[0], dtype=np.int8),
    ]
)

# %%
import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.5)
_ = plt.title("Training data")

# %%
rng = np.random.RandomState(0)
moons, _ = make_moons(n_samples=500, noise=0.05)
blobs, _ = make_blobs(
    n_samples=500, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25
)
outliers = rng.uniform(low=-3, high=3, size=(500, 2))
X_test = np.vstack([moons, blobs, outliers])
y_test = np.hstack(
    [
        np.ones(moons.shape[0], dtype=np.int8),
        np.zeros(blobs.shape[0], dtype=np.int8),
        rng.randint(0, 2, size=outliers.shape[0], dtype=np.int8),
    ]
)

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.5)
_ = plt.title("Testing data")

# %%
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)

# %%
from sklearn.inspection import DecisionBoundaryDisplay

_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    classifier, X_train, ax=ax, alpha=0.5, response_method="predict"
)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.5)

# %%
classifier.score(X_test, y_test)
_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    classifier, X_test, ax=ax, alpha=0.5, response_method="predict"
)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.5)

# %%
novelty_dectector = IsolationForest(n_estimators=100, random_state=0)
novelty_dectector.fit(X_train)
y_pred = novelty_dectector.predict(X_test)

# %%
X_test_inliers = X_test[y_pred == 1]
y_test_inliers = y_test[y_pred == 1]

# %%
classifier.score(X_test_inliers, y_test_inliers)
_, ax = plt.subplots()
DecisionBoundaryDisplay.from_estimator(
    classifier, X_test, ax=ax, alpha=0.5, response_method="predict"
)
ax.scatter(X_test_inliers[:, 0], X_test_inliers[:, 1], c=y_test_inliers, alpha=0.5)

# %%
