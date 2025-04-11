# Cargar paquetes:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth

# Configurar generalidades de la App:
st.set_page_config(page_title='Clustering techniques', page_icon='lee_logo.png')

# Cargar la información del dataFrame:
df = pd.read_csv('wholesale_customers_data.csv')

# Encabezado de la App:
left, middle, right = st.columns(3, vertical_alignment='bottom')
with middle:
    st.image('cluster.svg', width=150)
st.title('Clustering techniques.')

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
        st.subheader('Mean-Shift Clustering')

        # Slider para ajustar el quantile
        quantile_value = st.slider('Select the quantile value for bandwidth estimation:', 0.0, 1.0, 0.6, step=0.05)
        st.write(f'Selected quantile: **{quantile_value}**')

        # Estimar el ancho de banda
        bandwidth = estimate_bandwidth(df_var, quantile=quantile_value)
        st.write(f'Estimated bandwidth: **{bandwidth:.3f}**')

        # Aplicar Mean-Shift
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(df_var)
        labels = ms.labels_
        centroids = ms.cluster_centers_

        # Añadir etiquetas al dataframe
        df_var_clustered = df_var.copy()
        df_var_clustered['Group'] = labels

        # Visualización
        st.subheader('Cluster visualization:')
        fig_ms = sns.pairplot(data=df_var_clustered, hue='Group', palette='bright')
        st.pyplot(fig_ms)

main()

# Pie de página
st.markdown("""
<hr style='border:1px solid #ddd; margin-top: 40px; margin-bottom:10px'>
<div style='text-align: center; color: grey; font-size: 0.9em'>
    Developed by Edwin Lee | 📧 leonellee2016@gmail.com | Github: leonellee1988
</div>
""", unsafe_allow_html=True)