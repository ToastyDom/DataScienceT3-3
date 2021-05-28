import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
plt.style.use('seaborn-whitegrid')


# Preprocessing functions:
def preprocess_example_data(path):

    """Small function to preprocess example data
    There is probably an easier way

    Parameters:
    _______
    path: str
        link to txt

    Return:
    _______
    data: numpy array
        data in form [[x,y], ... ,[x,y]]
    """

    # Extract data from txt file
    with open(path, "r") as lines:
        raw_data = lines.readlines()
    x = []
    y = []
    for element in raw_data:
        element = element.split(" ")
        element = [x for x in element if x]  # remove whitespaces
        x.append(float(element[0]))
        y.append(float(element[1].rstrip("\n")))

    # Turn data into numpy array
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])
    data = np.array(data)
    return data


# Clustering Functions
def kmeans(data, amount_clusters):

    """K Means algorithm implementation

    Parameters
    ______
    data: numpy array
        array of datapoints
    clusters: int
        number of clusters we want to create

    Return
    ______
    centers: numpy array
        cluster centers
    labels: numpy array
        array that assigns each datapoint to a cluster"""

    # Fit data to kmeans model class
    kmeans = KMeans(n_clusters=amount_clusters, random_state=0).fit(data)

    # retrieve cluster centers and class labels to data
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # return back to API
    return centers, labels


def affinity(data):
    """Affinity Propagation implementation. This one does not require the amount of clusters

    Parameters
    ______
    data: numpy array
        array of datapoints

    Return
    ______
    centers: numpy array
        cluster centers
    labels: numpy array
        array that assigns each datapoint to a cluster"""

    # Initiate AfProp
    afprop = AffinityPropagation(random_state=5).fit(data)
    centers = afprop.cluster_centers_

    # predict clusters for data
    labels = afprop.predict(data)

    # return back to API
    return centers, labels


# Main functions:
def plotting(data, centers, labels):

    """Central plotting function that takes in all the relevant data to plot something

    Parameters:
    ______
    data: numpy array
        original datapoints
    centers: numpy array
        computed cluster centers
    labeles: numpy array
        computed classification to clusters"""

    # Ploting the data
    plt.scatter(data[:, 0], data[:, 1], c=labels,
                s=50, cmap='prism')
    plt.scatter(centers[:, 0], centers[:, 1], marker="+", color='blue')
    plt.show()


def centralAPI(algorithm, dataset, amount_clusters):

    """Central API that controls which algorithm we want to use and how we wish to configure them

    Parameters
    ______
    algorithm: str
        name of the cluster algorithm
    dataset: str
        name of dataset
    kwargs: xxx
        arguments that might depend on the clustering algorithm"""

    # Select and load dataset
    if dataset == "example":
        datapath = "./example_data.txt"
        data = preprocess_example_data(datapath)
    elif dataset == "iris":
        data = load_iris()["data"]
    elif dataset == "digits":
        digits = load_digits()
        data = digits.data
        target_label = digits.target
        # reduce dimensionality to make appropraite plots - if wanted
        # pca = PCA(n_components=2)
        # data = pca.fit_transform(digits.data)
    else:
        # TODO: Add new datasets and connect them elif statements
        pass

    # Select Algorithm
    if algorithm == "kmeans":
        centers, labels = kmeans(data, amount_clusters=amount_clusters)
    elif algorithm == "Affinity Propagation":
        centers, labels = affinity(data)
    else:
        # TODO: Add new algorithms and connect them with elif-statements
        pass

    # TODO: Guckt dass eure Algorithmen immer "centers" und "labels" returnen
    # Plot the data

    # PCS after kmeans leads to higher purity
    pca = PCA(n_components=2)
    data = pca.fit_transform(digits.data)
    # One plot with calculated labels and one with true labels to compare
    plotting(data, centers, labels)
    plotting(data, centers, target_label)

def purity(algorithm, dataset):
    """"Central function that calculates the external validation factor, done with "Purity"

    Parameters
    ______
    algorithm: str
        name of clustering algorithm
    dataset: str
        name of dataset
    """
    # We need to run another round of PCA should be handled through return of centralAPI 
    data, labels = load_digits(return_X_y=True)
    # Purity without PCA yields to better results
    # pca = PCA(n_components=2)
    # data = pca.fit_transform(data)
    _, predicted = kmeans(data, 10)

    # Calculate confusion Matrix which shows which points are in each cluster 
    # (predicted and should be)
    mat = confusion_matrix(labels, predicted)

    # normalizing over all clusters, therefore we do not need to multiply with 1/N
    # mat_norm is a matrix with i-th row = true label and j-th column = predicted label
    mat_norm = confusion_matrix(labels, predicted, normalize='all')

    # Calculate which predicted label matches to the true label
    # e.g. predicted label 1 is true label 9 if [_,9,_,...]
    mapping = np.array([np.argmax(mat[:,i]) for i in range(10)])
    mapping_norm = np.array([np.argmax(mat_norm[:,i]) for i in range(10)])
    
    # Calculate Purity 
    purity_value = 0
    for i in range(10):
        # mapping_norm[i] gives true label and i gives what was predicted
        purity_value += mat_norm[mapping_norm[i],i]
    print("Purity is: ", purity_value)

# Todo: Das müssen wir am Ende besser steuern. Das was wir hier aktuell eingeben wird später
#  unser Webinterface

"""Die algorithmen hier unten funktionieren bereits"""

# Choose from "kmeans", "Affinity Propagation"
algorithm = "kmeans"

# Choose from "example", "iris"
dataset = "iris"  # from machine learning 2
dataset_2 = "digits"
clusters = 5

# Auskommentieren, was man nicht ausführen möchte

# centralAPI(algorithm=algorithm, dataset=dataset, amount_clusters=clusters)
centralAPI(algorithm="kmeans", dataset=dataset_2, amount_clusters=10)
purity("kmeans", "")