# To start the frontend install streamlit and enter
# "streamlit hello" to start the webinterface at localhost
# Then run "streamlit run frontend.py"

import streamlit as st
from main import centralAPI

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
                      ('IRIS', 'Wine', '???'))

# select algorithm
sel_algorithm = st.selectbox('Which algorithm would you like to test?',
                      ('K-Means', 'Affinity Propagation', '???'))

# select amount of clusters if applicable
if sel_algorithm == "K-Means":
    cluster_amount = True
    sel_amount_cluster = st.text_input("Enter amount of clusters", 2)
    try:
        sel_amount_cluster = int(sel_amount_cluster)
        if sel_amount_cluster < 2:
            st.write("Enter integer above 1")
    except:
        st.write("Enter integer above 1")

# select evaluation algorithm
sel_evaluaton = st.selectbox('Which evaluation would you like to compute?',
                      ('Purity', '???', '???'))

# Setup summary
st.write('**Current setup:**')
st.write('Dataset:', sel_dataset)
st.write('Algorithm:', sel_algorithm)
st.write('Evaluation:', sel_evaluaton)



# Start evaluation
# Pressed button returns "True"
if st.button('Evaluate'):
    if cluster_amount:
        result = centralAPI(sel_algorithm, sel_dataset, sel_amount_cluster)
    else:
        result = centralAPI(sel_algorithm, sel_dataset, amount_clusters=None)

    # Todo: Vielleicht sollten wir bei der central API als return die plots haben damit die hier direkt
    # Todo: zu erreichen sind. Diese dann auf dem Bildschrim darstellen und dann passt das denke ich!
    st.write('Computing...')
    st.write(result)
