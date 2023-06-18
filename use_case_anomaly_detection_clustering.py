# %% [markdown]
#
# # Customer segmentation use case
#
# This example shows a use case where we are interested at segmenting customers.
# Customer segmentation is principally useful for marketing strategies.
#
# We use the UCI dataset available at:
# https://archive.ics.uci.edu/dataset/352/online+retail
#
# This is a transnational data set which contains all the transactions
# occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered
# non-store online retail.
#
# The following of this use-case is subdivided into 2 parts. At first, we focus
# on analyzing the dataset and show how to detect outliers using on of the
# outlier detection algorithm. Subsequently, we will preprocess our dataset to
# extract marketing relevant features (Recency, Frequency, Monetary) and
# use a clustering algorithm to segment our customers.
#
# ## Outlier detection
#
# Let's first load the dataset and look at what we got.

# %%
import pandas as pd

df = pd.read_excel("data/online_retail.xlsx")
df = df.convert_dtypes()

# %%
df.head()

# %%
df.info()

# %% [markdown]
#
# From the information above, we see that we have ~550k samples and 8 features. A
# sample corresponds to a customer transaction. It could be noted that a customer
# can have multiple transactions:

# %%
df["CustomerID"].value_counts()

# %% [markdown]
#
# From the information above, we see that we have missing values in some of the
# features, notably in the "CustomerID" feature. Since, we have a large number of
# samples, we drop the samples with missing values.

# %%
df = df.dropna(axis=0)
df.info()

# %% [markdown]
#
# Therefore, we have ~400k samples left. Let's look at the distribution of the
# features. In terms of numeric features, we particularly interested in the "UnitPrice"
# and "Quantity" features that will be used later for the customer segmentation.

# %%
import matplotlib.pyplot as plt

plt.hist(df["Quantity"], bins=20)
plt.xlabel("Quantity")
plt.ylabel("Frequency")
_ = plt.title("Distribution of the Quantity feature")

# %%
plt.hist(df["UnitPrice"], bins=20)
plt.xlabel("Unit price")
plt.ylabel("Frequency")
_ = plt.title("Distribution of the UnitPrice feature")

# %% [markdown]
#
# We observe that the range of both features is very large while all the samples are
# concentrated in a few bins. This is typical case that some extreme values are present
# in the dataset. Let's use an outlier detection algorithm to automatically detect such
# samples.

# %%
from sklearn.ensemble import IsolationForest

model = IsolationForest(n_estimators=400)
y_pred = model.fit_predict(df[["Quantity", "UnitPrice"]])
y_pred

# %%
from collections import Counter

Counter(y_pred)

# %% [markdown]
#
# We used `IsolationForest` to detect the outliers. Looking at the predictions of this
# estimator, we see that it provides 2 types of output: 1 for inliers and -1 for
# outliers. It means that for our further processing, we use only the inliers.
# We will have a closer look at the distribution of the previous features for the
# inliers and outliers to see what type of data are detected by the algorithm.

# %%
inliers = df[y_pred == 1]
outliers = df[y_pred == -1]

# %%
_, axs = plt.subplots(ncols=2, figsize=(12, 4))

axs[0].hist(inliers["Quantity"], bins=20)
axs[0].set_xlabel("Quantity")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Distribution of the Quantity feature for inliers")

axs[1].hist(inliers["UnitPrice"], bins=20)
axs[1].set_xlabel("Unit price")
axs[1].set_ylabel("Frequency")
_ = axs[1].set_title("Distribution of the UnitPrice feature for outliers")

# %% [markdown]
#
# Now, we observe that we have a more reasonable distribution for the inliers.
# Without any surprised, the extreme values are detected as outliers:

# %%
outliers[["Quantity", "UnitPrice"]].describe()

# %% [markdown]
#
# So the dataset that we will use for the customer segmentation is the following:

# %%
inliers.info()

# %% [markdown]
#
# ## Customer segmentation
#
# Now, we use the inliers to extract marketing relevant features and use a clustering
# algorithm to segment our customers. In this regard, we use the Recency, Frequency,
# Monetary (RFM) model.
#
# The Recency feature corresponds to the number of days since the last transaction of a
# customer. The lower the recency, the more recent the transaction is and the more
# likely the customer is active. Let's compute this feature and store it in a dataframe.
#
# Since a customer will have several transactions, we will consider the latest
# transaction.


# %%
reference_date = inliers["InvoiceDate"].max()
recency = (reference_date - inliers["InvoiceDate"]).to_frame()
recency["CustomerID"] = inliers["CustomerID"]
recency = recency.groupby("CustomerID").min()
recency = recency.rename(columns={"InvoiceDate": "Recency"})
recency["Recency"] = recency["Recency"].dt.days
recency

# %% [markdown]
#
# The Frequency feature corresponds to the number of transactions of a customer. The
# higher the frequency, the more likely the customer is active. Let's compute this
# feature and store it in a dataframe.

# %%
frequency = (
    inliers.groupby(["CustomerID", "InvoiceDate"]).count().groupby("CustomerID").count()
)
frequency = frequency[["InvoiceNo"]].rename(columns={"InvoiceNo": "Frequency"})
frequency

# %% [markdown]
#
# The Monetary feature corresponds to the total amount spent by a customer. The higher
# the monetary, the more likely the customer is active. Let's compute this feature and
# store it in a dataframe.

# %%
monetary = inliers[["CustomerID", "Quantity", "UnitPrice"]].copy()
monetary["Monetary"] = monetary["Quantity"] * monetary["UnitPrice"]
monetary = monetary.drop(columns=["Quantity", "UnitPrice"])
monetary = monetary.groupby("CustomerID").sum()
monetary

# %% [markdown]
#
# Now, we merge all 3 features together in a single dataframe.

# %%
rfm = pd.concat([recency, frequency, monetary], axis=1)
rfm.index = rfm.index.astype(int)
rfm

# %% [markdown]
#
# At this stage, we are ready to use a clustering algorithm to segment our customers.
# Regrouping the customers in cluster allow us to apply different strategy depending on
# the cluster characteristics. For instance, we can target the customers that are
# inactive for a long time with a special offer to try to reactivate them.
#
# We use the `KMeans` algorithm to perform the clustering. However, there is two
# important things to consider before analyzing the results of the clustering.
#
# First, we should look at the scale of our features:

# %%
rfm.describe()

# %% [markdown]
#
# We observe that we have different range for the three features. `KMeans` will use
# an Euclidean distance to compute the distance between two samples. Therefore, the
# distance will be dominated by the "Monetary" feature because is dynamic range is
# much larger. If we want each feature to contribute equally to the clustering, we
# should therefore scale the features to the same range before applying the clustering.
#
# Then, we should also select the number of clusters to use. We will try to apply the
# elbow method and check where to we go from there. It means that we will fit several
# `KMeans` model and look at the inertia to select an appropriate number of clusters.
#
# Let's start by creating a pipeline that scale the data and then cluster it:

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

cluster = Pipeline(steps=[
    ("scaler", StandardScaler()), ("kmeans", KMeans(n_init=10))
])
cluster

# %% [markdown]
#
# Now, we cluster the dataset with a different number of clusters and store the inertia.

# %%
inertia = {}
for n_clusters in range(1, 20):
    cluster.set_params(kmeans__n_clusters=n_clusters)
    cluster.fit(rfm)
    inertia[n_clusters] = cluster[-1].inertia_

# %% [markdown]
#
# Finally, let's check the inertia for each number of clusters:

# %%
_, ax = plt.subplots()
ax.plot(list(inertia.keys()), list(inertia.values()), marker="o")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Inertia")
_ = ax.set_xticks(range(1, 20))

# %% [markdown]
#
# It is not super obvious which number of cluster to select from this plot. A number
# between 4 and 7 seems reasonable. We can first have a try with 6 clusters and check
# the results.

# %%
cluster.set_params(kmeans__n_clusters=6).fit(rfm)
prototype_customer = pd.DataFrame(
    cluster[0].inverse_transform(cluster[-1].cluster_centers_),
    columns=rfm.columns,
)
prototype_customer["Number of customer"] = pd.Series(Counter(cluster[-1].labels_))
prototype_customer

# %% [markdown]
#
# We observe that with 6 clusters, we have 2 clusters with very few customers.
# Therefore, we reduce the number of clusters to 4.

# %%
cluster.set_params(kmeans__n_clusters=4).fit(rfm)
prototype_customer = pd.DataFrame(
    cluster[0].inverse_transform(cluster[-1].cluster_centers_),
    columns=rfm.columns,
)
prototype_customer["Number of customer"] = pd.Series(Counter(cluster[-1].labels_))
prototype_customer

# %% [markdown]
#
# The cluster #2 corresponds to our most active customers. The recency is really low and
# the amount spent is really high. The cluster #3 are probably the most important
# customers: they are active, spend a lot money and account for ~7% of the sells.
# If we consider the Pareto principle, we should focus on the cluster #1, #2 and #3
# since they would account for a large part of the sells and account for 30% of the
# customers.
