import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation
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

    #return back to API
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
    P = afprop.predict(data)

    # return back to API
    return centers, P





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



def centralAPI(algorithm, datapath, amount_clusters):

    """Central API that controls which algorithm we want to use and how we wish to configure them

    Parameters
    ______
    algorithm: str
        name of the cluster algorithm
    datapath: str
        path to the dataset
    kwargs: xxx
        arguments that might depend on the clustering algorithm"""


    if algorithm == "kmeans":
        data = preprocess_example_data(datapath)
        centers, labels = kmeans(data, amount_clusters= amount_clusters)
    elif algorithm == "Affinity Propagation":
        data = preprocess_example_data(datapath)
        centers, labels = affinity(data)
    else:
        # TODO: Add new algorithms and connect them with elif-statements
        pass


    # TODO: Guckt dass eure Algorithmen immer "centers" und "labels" returnen
    # Plot the data
    plotting(data, centers, labels)








# Todo: Das müssen wir am Ende besser steuern. Das was wir hier aktuell eingeben wird später
#  unser Webinterface

"""Die algorithmen hier unten funktionieren bereits"""

# algorithm = "kmeans"
algorithm = "Affinity Propagation"
datapath = "./example_data.txt"  # from machine learning 2
clusters = 5

centralAPI(algorithm = algorithm, datapath = datapath, amount_clusters=clusters)