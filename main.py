import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, Birch
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

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


def gaussian_mixture_model(data, amount_clusters):
    """ Gaussian mixture model implementation. Requires the amount of clusters.

    Parameters
    ______
    data: numpy array
        array of datapoints


    Return
    ______
    gmm: gaussian mixture object
        contains information about gaussian distributions in dataset
    labels: numpy array
        array that assigns each datapoint to a cluster"""

    # Create gaussian mixture model
    gmm = GaussianMixture(n_components=amount_clusters).fit(data)

    # predict clusters for data
    labels = gmm.predict(data)

    return data, False, labels




def birch(data, amount_clusters):
    """Birch implementation. 
    amount_clusters: int
        number of clusters we want to create

    Return
    ______
    centers: numpy array
        cluster centers
    labels: numpy array
        array that assigns each datapoint to a cluster"""
        
    # Initiate BIRCH
    birch_fit = Birch(threshold=0.01, n_clusters=amount_clusters).fit(data)
    centers = birch_fit.subcluster_centers_
    
    # predict clusters for data
    labels = birch_fit.predict(data)
    
    return centers, labels



def plotting(data, labels):

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
    #plt.scatter(centers[:, 0], centers[:, 1], marker="+", color='blue')
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

    elif dataset == "IRIS":
        data_obj = load_iris()

    elif dataset == "digits":
        data_obj = load_digits()

    elif dataset == "breast_cancer":
        data_obj = load_breast_cancer()

    elif dataset == "Wine":
        data_obj = load_wine()

    else:
        print("Invalid input!")
        return

    data = data_obj.data
    target_labels = data_obj.target

    # Select Algorithm
    if algorithm == "K-Means":
        centers, labels = kmeans(data, amount_clusters=amount_clusters)

    elif algorithm == "Affinity Propagation":
        centers, labels = affinity(data)

    elif algorithm == "Gaussian mixture model":
        data, gmm, labels = gaussian_mixture_model(data, amount_clusters=amount_clusters)

    elif algorithm == "BIRCH":
        centers, labels = birch(data, amount_clusters=amount_clusters)

    else:
        print("Invalid Input!")
        return


    # TODO: Guckt dass eure Algorithmen immer "centers" und "labels" returnen
    
    # Calculate purity before plotting 
    pur_val = purity(labels, target_labels)

    # Plot the data
    # Use PCA to enable plotting high dimensional data in 2d
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)

    # One plot with calculated labels and one with true labels to compare
    plotting(data, labels)
    #plotting(data, centers, target_labels)
  
    return data, labels, pur_val
    

def purity(labels, targets):
    """"Central function that calculates the external validation factor, done with "Purity"

    Parameters
    ______
    data: str (?)
        calculated data from the algorithms
    labels: str (?)
        target labels to compute purity
    amount: int
        number of clusters for k-means
    """
    # We need to run another round of PCA should be handled through return of centralAPI 
    # data, labels = load_digits(return_X_y=True)
    # Purity without PCA yields to better results
    # pca = PCA(n_components=2)
    # data = pca.fit_transform(data)

    # calculate amount of clusters
    amount = len(set(labels))

    # Calculate confusion Matrix which shows which points are in each cluster 
    # (predicted and should be)
    mat = confusion_matrix(targets, labels)

    # normalizing over all clusters, therefore we do not need to multiply with 1/N
    # mat_norm is a matrix with i-th row = true label and j-th column = predicted label
    mat_norm = confusion_matrix(targets, labels, normalize='all')

    # Calculate which predicted label matches to the true label
    # e.g. predicted label 1 is true label 9 if [_,9,_,...]
    mapping = np.array([np.argmax(mat[:, i]) for i in range(amount)])
    mapping_norm = np.array([np.argmax(mat_norm[:, i]) for i in range(amount)])
    
    # Calculate Purity 
    purity_value = 0
    for i in range(amount):
        # mapping_norm[i] gives true label and i gives what was predicted
        purity_value += mat_norm[mapping_norm[i],i]
    print("Purity is: ", purity_value)
    return purity_value

# Todo: Das müssen wir am Ende besser steuern. Das was wir hier aktuell eingeben wird später
#  unser Webinterface

"""Die algorithmen hier unten funktionieren bereits"""

# Choose from "example", "iris", beast_cancer
datasets = ["IRIS", "Wine", "digits", "breast_cancer"]

clusters = 5

# Auskommentieren, was man nicht ausführen möchte

algorithms = ["K-Means", "Affinity Propagation", "Gaussian mixture model", "BIRCH"]
centralAPI(algorithm=algorithms[1], dataset=datasets[0], amount_clusters=3)
"""
for i in range(4):
    for j in range(4):
        print(algorithms[i], " ", datasets[j])
        centralAPI(algorithm=algorithms[i], dataset=datasets[j], amount_clusters=3)
"""