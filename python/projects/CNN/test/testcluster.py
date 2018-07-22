"""
Clustering elements using K-Means algorithm
Executing data visualization using pyplot too
More info to use this elements
https://matplotlib.org/users/pyplot_tutorial.html
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def main():
    print('Initializing clustering from elements in list')
    print('Generating elements in list')

    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(kmeans.labels_)

    predictions = kmeans.predict([[0, 0], [4, 4]])
    print(predictions)

    print('Printing cluster centers')
    print(kmeans.cluster_centers_)

    print('Plotting data using pyplot')
    plt.plot(X[:, 0], X[:, 1], 'ro')
    plt.show()

    print('Done!')


if __name__ == '__main__':
    main()
