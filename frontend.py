# To start the frontend install streamlit and enter
# "streamlit hello" to start the webinterface at localhost
# Then run "streamlit run frontend.py"

import streamlit as st
from main import centralAPI
import matplotlib.pyplot as plt

# Setup
st.title('Evalation of Clustering-Algorithms')
st.subheader("Project Group: T3-3")
cluster_amount = False


# Descripton - Just a draft
st.write("""
Clustering is still a hot topic. With the growth of large data sets with multiple dimensions, \
the question of the right clustering algorithm becomes more and more relevant. \
"Which algorithm is best suited for my data set?". This is the question we are dealing with.
With this tool you have the possibility to test 4 different clustering algorithms \
on 4 different datasets according to their goodness. The goal is to show that depending \
on the dataset and data arrangement, a different algorithm can yield certain advantages.
""")


# Interface
st.title("Setup Evaluation")

# select dataset
sel_dataset = st.selectbox('Which dataset would you like to test?',
                      ("IRIS", "Wine", "Digits", "Breast Cancer"))

# select algorithm
sel_algorithm = st.selectbox('Which algorithm would you like to test?',
                      ("K-Means", "Affinity Propagation", "Gaussian mixture model", "BIRCH"))

# select amount of clusters if applicable
if (sel_algorithm == "K-Means") or (sel_algorithm == "Gaussian mixture model")\
    or (sel_algorithm == "BIRCH"):
    cluster_amount = True
    if sel_dataset == "IRIS":
        sel_amount_cluster = st.text_input("Enter amount of clusters", 3)
    elif sel_dataset == "Wine":
        sel_amount_cluster = st.text_input("Enter amount of clusters", 3)
    elif sel_dataset == "Digits":
        sel_amount_cluster = st.text_input("Enter amount of clusters", 10)
    else:
        sel_amount_cluster = st.text_input("Enter amount of clusters", 2)
    try:
        sel_amount_cluster = int(sel_amount_cluster)
        if sel_amount_cluster < 2:
            st.write("Enter integer above 1")
    except:
        st.write("Enter integer above 1")

# select evaluation algorithm
#sel_evaluaton = st.selectbox('Which evaluation would you like to compute?',
#                      ('Purity', '???', '???'))

# Setup summary
st.write('**Current setup:**')
st.write('Dataset:', sel_dataset)
st.write('Algorithm:', sel_algorithm)
st.write('Evaluation:', "Purity")



# Start evaluation
# Pressed button returns "True"
if st.button('Evaluate'):
    if cluster_amount:
        data, labels, purity_val, tar_labels = centralAPI(sel_algorithm, sel_dataset, sel_amount_cluster)
    else:
        data, labels, purity_val, tar_labels = centralAPI(sel_algorithm, sel_dataset, amount_clusters=None)

    # Todo: Vielleicht sollten wir bei der central API als return die plots haben damit die hier direkt
    # Todo: zu erreichen sind. Diese dann auf dem Bildschrim darstellen und dann passt das denke ich!
    
    # Need Figure for st.pyplot

    """
    # Side by Side Version
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    ax1.set_title("Label predictions by algorithm")
    ax2.set_title("Original labels of data set")
    ax1.scatter(data[:, 0], data[:, 1], c=labels,
                s=50, cmap='prism')
    ax2.scatter(data[:, 0], data[:, 1], c=tar_labels,
                s=50, cmap='prism')
    st.pyplot(fig)
    # Side by Side Version End
    """
    
    
    # Below Version Start
    figb, axb = plt.subplots()
    axb.scatter(data[:, 0], data[:, 1], c=tar_labels,
                s=50, cmap='prism')
    axb.set_title("Original labels of data set")
    st.pyplot(figb)
    
    fig, ax = plt.subplots()
    axb.scatter(data[:, 0], data[:, 1], c=labels,
                s=50, cmap='prism') 
    axb.set_title("Label predictions by algorithm")
    st.pyplot(figb)
    # Below Version End

    
    st.write("The purity value of ", sel_algorithm, " on ", \
             "'" + sel_dataset + "'", " is: ", purity_val)

