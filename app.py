# Cargar paquetes:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar generalidades de la App:
st.set_page_config(page_title='Clustering techniques', page_icon='lee_logo.png')

# Cargar la información del dataFrame:
df = pd.read_csv('wholesale_customers_data.csv')

# Encabezado de la App:
st.image('cluster.svg', width=300)
st.title('Clustering techniques')

def main():

    # Mostrar Dataframe:
    st.subheader(f'Dataframe: Customer purchases by product category')
    st.dataframe(df)

main()

# Pie de página
st.markdown("""
<hr style='border:1px solid #ddd; margin-top: 40px; margin-bottom:10px'>
<div style='text-align: center; color: grey; font-size: 0.9em'>
    Developed by Edwin Lee | 📧 leonellee2016@gmail.com | Github: leonellee1988
</div>
""", unsafe_allow_html=True)