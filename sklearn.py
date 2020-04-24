from matplotlib.cbook import maxdict
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## Q1
#Use a linear regression model with the Boston housing data set. 
# return which factor has the largest effect on the price of housing in Boston. 
boston = load_boston()
x = boston.data
y = boston.target
model = LinearRegression()
model.fit(x, y)
print(model.coef_)

maxeffect = max(model.coef_)
for i in range(len(model.coef_)):
    if model.coef_[i] == maxeffect:
        print("largest effect on housing price in Boston:"+str(i))


## Q2
# Use a KMeans regression model with the Iris data set. 
# Graph the fit when using differing numbers of clusters. 
# Graph the result and either corroborate or refute the assumption that the data set represents 3 different varieties of iris.

iris = datasets.load_iris()  #loading data
X = iris.data[:, :2] # define target
y = iris.target  # define predictors

# scatter plot of original dataset
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Spea1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)


# k=2
km_2 = KMeans(n_clusters = 2, n_jobs = 4, random_state=20)
km_2.fit(X)
centers2 = km_2.cluster_centers_

# k=3
km_3 = KMeans(n_clusters = 3, n_jobs = 4, random_state=20)
km_3.fit(X)
centers3 = km_3.cluster_centers_


# k=4
km_4 = KMeans(n_clusters = 4, n_jobs = 4, random_state=20)
km_4.fit(X)
centers4 = km_4.cluster_centers_

print(centers2,centers3,centers4)


#plot k=2, k=3, k=4
fig, axes = plt.subplots(1, 3, figsize=(16,8)) # plot three plot in one line

axes[0].scatter(X[:, 0], X[:, 1], c=km_2.labels_, cmap='jet',edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('2 cluster', fontsize=18)

axes[1].scatter(X[:, 0], X[:, 1], c=km_3.labels_, cmap='jet',edgecolor='k', s=150)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].set_title('3 cluster', fontsize=18)

axes[2].scatter(X[:, 0], X[:, 1], c=km_4.labels_, cmap='jet',edgecolor='k', s=150)
axes[2].set_xlabel('Sepal length', fontsize=18)
axes[2].set_ylabel('Sepal width', fontsize=18)
axes[2].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[2].set_title('4 cluster', fontsize=18)
