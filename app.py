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
left, middle, right = st.columns(3, vertical_alignment='bottom')
with middle:
    st.image('cluster.svg', width=150)
st.title('Clustering techniques.')

def main():

    # Mostrar dataframe:
    st.subheader(f'Dataframe: Customer purchases by product category (first 5 facts).')
    st.dataframe(df.head(5))

    # Crear matriz de correlación y distribución:
    st.subheader(f'Correlation and distribution matrix.')
    fig = sns.pairplot(data=df)
    st.pyplot(fig)

    # Generación del nuevo dataframe con las variables de interés:
    st.subheader(f'Generating the new dataframe with the variables: "Grocery" and "Detergents_paper".')
    df_var = df[['Grocery','Detergents_Paper']]
    st.dataframe(df_var.head(5))
    st.write('''
            Note:
            1) These variables were chosen because they exhibit a positive correlation.
            2) The first 5 data points from the new data frame are shown.
            d''')

main()

# Pie de página
st.markdown("""
<hr style='border:1px solid #ddd; margin-top: 40px; margin-bottom:10px'>
<div style='text-align: center; color: grey; font-size: 0.9em'>
    Developed by Edwin Lee | 📧 leonellee2016@gmail.com | Github: leonellee1988
</div>
""", unsafe_allow_html=True)