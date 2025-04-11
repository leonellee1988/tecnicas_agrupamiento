# Cargar paquetes:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# Configurar generalidades de la App:
st.set_page_config(page_title='Clustering techniques', page_icon='lee_logo.png')

# Cargar la información del dataFrame:
df = pd.read_csv('wholesale_customers_data.csv')

# Encabezado de la App:
left, middle, right = st.columns(3, vertical_alignment='bottom')
with middle:
    st.image('cluster.svg', width=150)
st.title('Clustering techniques.')

# Función para el método Mean-Shift:
def mean_shift_clustering(data):
    st.subheader('Mean-Shift Clustering')
    quantile_value = st.slider('Select the quantile value for bandwidth estimation:', 0.0, 1.0, 0.6, step=0.05)
    bandwidth = estimate_bandwidth(data, quantile=quantile_value)
    st.write(f'Estimated bandwidth: **{bandwidth:.3f}**')

    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(data)
    labels = ms.labels_

    result = data.copy()
    result['Group'] = labels

    st.subheader('Cluster visualization:')
    fig = sns.pairplot(result, hue='Group', palette='bright')
    st.pyplot(fig)

from sklearn.cluster import KMeans

# Función para el método KMeans:
def kmeans_clustering(data):
    st.subheader('KMeans Clustering')

    # Método de codo (Elbow method)
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=1234, n_init='auto')
        kmeans.fit(data)
        inertia.append((i, kmeans.inertia_))

    fig_elbow, ax = plt.subplots()
    ax.plot([x[0] for x in inertia], [x[1] for x in inertia], marker="X")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig_elbow)

    # Número de clústeres elegido por el usuario
    clusters = st.number_input('Choose the number of clusters based on the elbow plot:', min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=clusters, random_state=1234, n_init='auto')
    labels = kmeans.fit_predict(data)

    result = data.copy()
    result['Group'] = labels

    st.subheader('Cluster visualization:')
    fig = sns.pairplot(result, hue='Group', palette='bright')
    st.pyplot(fig)

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# Función para el método Agglomerative:
def agglomerative_clustering(data, full_df):
    st.subheader('Agglomerative Clustering')

    # Dendrograma
    linked = linkage(full_df, method='ward', metric='euclidean')
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=True, p=30, ax=ax)
    plt.xticks([])
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Euclidean Distance')
    st.pyplot(fig)

    # Número de clústeres elegido por el usuario
    n_clusters = st.number_input('Choose the number of clusters based on the dendrogram:', min_value=2, max_value=10, value=3)

    cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = cluster.fit_predict(full_df)

    result = data.copy()
    result['Cluster'] = labels

    st.subheader('Cluster visualization:')
    fig = sns.pairplot(result, hue='Cluster', palette='bright')
    st.pyplot(fig)

from sklearn.cluster import DBSCAN

# Función método DBSCAN:
from sklearn.cluster import DBSCAN

def dbscan_clustering(data):
    st.subheader('DBSCAN Clustering')

    # Aplicar DBSCAN sin parámetros ajustables
    dbscan = DBSCAN()
    dbscan.fit(data)
    labels = dbscan.labels_

    # Añadir resultados al DataFrame
    result = data.copy()
    result['Group'] = labels

    # Visualización
    st.subheader('Cluster visualization:')
    fig = sns.pairplot(result, hue='Group', palette='bright')
    st.pyplot(fig)

    # Mostrar número de grupos encontrados (excluyendo ruido -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    st.write(f'Number of clusters found: **{n_clusters}**')
    if -1 in labels:
        st.write('Note: Some points were labeled as noise (`-1`) by DBSCAN.')

# Función principal:
def main():

    # Mostrar dataframe:
    st.subheader(f'Dataframe: Customer purchases by product category.')
    st.dataframe(df.head(5))
    st.write('''
            **Note:**
            1) The first 5 data points from the dataframe are shown.
            ''')

   # Crear matriz de correlación y mapa de calor
    st.subheader('Correlation heatmap between product categories')
    # Calcular la matriz de correlación
    corr_matrix = df.corr()
    # Crear figura del mapa de calor
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)

    # Mostrar en Streamlit
    st.pyplot(fig)

    # Generación del nuevo dataframe con las variables de interés:
    st.subheader(f'Generating the new dataframe with the variables: "Grocery" and "Detergents_paper".')
    df_var = df[['Grocery','Detergents_Paper']]
    st.dataframe(df_var.head(5))
    st.write('''
            **Note:**
            1) These variables were chosen because they exhibit a positive correlation.
            2) The first 5 data points from the new dataframe are shown.
            ''')

    # Selección del tipo de técnica de agrupamiento
    st.subheader('Choose the type of grouping technique:')
    
    # Menú para seleccionar el método de agrupación:
    clustering_method = st.selectbox(
        'Select a clustering algorithm:',
        ('', 'Mean-Shift', 'KMeans', 'Agglomerative', 'DBSCAN')
    )
    
    if clustering_method == 'Mean-Shift':
        mean_shift_clustering(df_var)

    elif clustering_method == 'KMeans':
        kmeans_clustering(df_var)

    elif clustering_method == 'Agglomerative':
        agglomerative_clustering(df_var, df)

    elif clustering_method == 'DBSCAN':
        dbscan_clustering(df_var)

main()

# Pie de página
st.markdown("""
<hr style='border:1px solid #ddd; margin-top: 40px; margin-bottom:10px'>
<div style='text-align: center; color: grey; font-size: 0.9em'>
    Developed by Edwin Lee | 📧 leonellee2016@gmail.com | Github: leonellee1988
</div>
""", unsafe_allow_html=True)