# Cargar paquetes:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar generalidades de la App:
st.set_page_config(page_title='Clustering techniques', page_icon='lee_logo.png')

# Cargar la información del dataFrame:
df = pd.read_excel('wholesale_customers_data.csv')

# Encabezado de la App:
st.image('cluster.svg', width=150)
st.title('Clustering techniques')

def main():

    # Mostrar Dataframe:
    st.dataframe(df)