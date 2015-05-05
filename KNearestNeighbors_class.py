###########################################
#! k Nearest Neighbors Individual sprint !#
###########################################




import numpy as np
import sklearn as sk
# import my_KNearestNeighbors
from numpy import tile
import operator
from sklearn.datasets import make_classification
from scipy.spatial.distance import euclidean, cosine
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

def make_fake_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, class_sep=5, random_state=5)
    return X, y


# def my_KNearestNeighbors(centroid, dataSet, labels, k = 3, distance = 'euclidean_distance'):

    # dataSetSize = dataSet.shape[0]
    # diffMat = np.subtract( tile(centroid, (dataSetSize,1)) , dataSet)
    # sqDiffMat = diffMat**2 
    # sqDistances = sqDiffMat.sum(axis=1) 
    # distances = sqDistances**0.5 
    # sortedDistIndicies = distances.argsort()

    # classCount={} 
    # for i in range(k):     
    #     voteIlabel = labels[sortedDistIndicies[i]] 
    #     classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    #     sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) 
    # return sortedClassCount[0][0]

class KNearestNeighbors():

    def __init__(self, k, distance, X, y):
        self.k = k
        self.distance = distance
        self.X = np.array(X)
        self.y = np.array(y)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X, y, distance):
        y_pred = []
        for i in xrange(len(self.y)):
            centroid = self.X[i]
            diffMat = distance(centroid, X)
            sqDiffMat = diffMat**2 
            sqDistances = sqDiffMat.sum(axis=1) 
            distances = sqDistances**0.5 
            sortedDistIndicies = distances.argsort()

            classCount={} 
            for i in range(k):     
                voteIlabel = labels[sortedDistIndicies[i]] 
                classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
                sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) 
                y_pred.append(sortedClassCount[0][0])
        return y_pred    


def scoreKNN(y_pred, labels):
    correct = 0
    for i,j in zip(y_pred,labels): 
        if i == j:
            correct += 1
    return float(correct)/ len(labels)

def main():
    # X, y = make_fake_data()

    # Iris Data (2d only)
    # n_neighbors = 15

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features. We could
                          # avoid this ugly slicing by using a two-dim dataset
    y = iris.target

    h = .02  # step size in the mesh
    for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
        clf = KNearestNeighbors(k = 3, distance = 'euclidean', X = X, y = y)
        clf.fit(X,y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(xx.ravel(), yy.ravel(), 'euclidean')

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

    plt.show()



if __name__ == "__main__":
    main()    
