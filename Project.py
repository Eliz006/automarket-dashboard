import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Función para cargar y limpiar los datos
@st.cache_data
def load_data():
    df = pd.read_csv("car_prices.csv")
    df = df.drop(columns=['transmission', 'trim', 'vin', 'interior', 'seller', 'saledate', 'condition'])
    df['body'] = df['body'].str.lower().replace({'suv': 'SUV'})
    df_clean = df.dropna(subset=['sellingprice', 'year', 'odometer', 'mmr', 'make', 'model', 'body'])
    return df_clean

# Carga de datos
df_clean = load_data()
df_sample = df_clean.sample(5000, random_state=42) if len(df_clean) > 5000 else df_clean

# Logo y título
st.image("logo.png", width=150)
st.title("Análisis de Autos Usados - AutoMarket USA")

# Pestañas
tab1, tab2, tab3 = st.tabs(["Datos", "Análisis Nacional", "Análisis Estatal"])

# TAB 1: DATOS
with tab1:
    st.header("Resumen de Datos")

    with st.expander("Estadísticas descriptivas"):
        st.write(df_clean.describe(include='all'))

    with st.expander("Mapa de calor de correlación"):
        corr = df_clean.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr, text_auto=True, title='Mapa de Calor de Correlación entre Variables Numéricas')
        st.plotly_chart(fig)

    with st.expander("Relaciones entre variables numéricas"):
        pairplot_path = "pairplot.png"
        sns.pairplot(df_clean[['sellingprice', 'mmr', 'odometer', 'year']])
        plt.suptitle("Relaciones entre variables numéricas", y=1.02)
        plt.savefig(pairplot_path)
        plt.close()
        st.image(pairplot_path, caption="Relaciones entre variables numéricas", use_container_width=True)
        if os.path.exists(pairplot_path):
            os.remove(pairplot_path)

# === TAB 2: ANÁLISIS NACIONAL ===
with tab2:
    st.header("Análisis Nacional")

    top_make = df_clean['make'].mode()[0]
    avg_price = df_clean['sellingprice'].mean()
    st.markdown(f"**Marca más comprada:** {top_make}")
    st.markdown(f"**Precio promedio de venta:** ${avg_price:,.2f}")

    with st.expander("Cantidad de Autos por Marca"):
        count_by_make = df_clean['make'].value_counts().reset_index()
        count_by_make.columns = ['make', 'count']
        fig_count = px.bar(count_by_make, x='make', y='count', color='make', title='Distribución de Cantidad de Autos por Marca')
        st.plotly_chart(fig_count)

    with st.expander("Odómetro vs Precio de Venta"):
        fig2 = px.scatter(df_sample, x='odometer', y='sellingprice', opacity=0.5, labels={'odometer': 'Odómetro', 'sellingprice': 'Precio de Venta'})
        st.plotly_chart(fig2)

    with st.expander("Distribución de Precios por Marca"):
        fig3 = px.box(df_sample, x='make', y='sellingprice', title='Distribución de Precios por Marca')
        st.plotly_chart(fig3)

    with st.expander("Regresión Lineal: MMR vs Precio de Venta"):
        @st.cache_data
        def entrenar_modelo(data):
            X = data[['mmr']]
            y = data['sellingprice']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            return X_test, y_test, y_pred, r2, mse

        X_test, y_test, y_pred, r2, mse = entrenar_modelo(df_clean)
        fig4 = px.scatter(x=X_test['mmr'], y=y_test, labels={'x': 'MMR', 'y': 'Precio de Venta'})
        fig4.add_traces(go.Scatter(x=X_test['mmr'], y=y_pred, mode='lines', name='Predicción', line=dict(color='red')))
        st.plotly_chart(fig4)
        st.markdown(f"**R²:** {r2:.4f} | **MSE:** {mse:,.2f}")

    with st.expander("Relación 3D entre MMR, Precio y Año"):
        try:
            fig5 = px.scatter_3d(df_sample, x='mmr', y='sellingprice', z='year', color='sellingprice', opacity=0.6)
            st.plotly_chart(fig5)
        except Exception:
            st.warning("No se pudo cargar el gráfico 3D correctamente.")

    with st.expander("Distribución del Precio por Año de Fabricación"):
        fig_box_year = px.box(df_sample, x='year', y='sellingprice', title="Boxplot: Precio vs Año del Vehículo")
        st.plotly_chart(fig_box_year)

# === TAB 3: ANÁLISIS ESTATAL ===
with tab3:
    st.header("Análisis Estatal")

    with st.expander("Total de Autos Vendidos por Marca en Cada Estado"):
        top_makes = df_clean['make'].value_counts().nlargest(10).index
        stacked_counts = df_clean[df_clean['make'].isin(top_makes)].groupby(['state', 'make']).size().reset_index(name='count')
        fig8 = px.bar(stacked_counts, x='state', y='count', color='make', title="Total de Autos Vendidos por Marca en Cada Estado", barmode='stack')
        st.plotly_chart(fig8)

    with st.expander("Distribución de Tipos de Carrocería por Estado"):
        top_bodies = df_clean['body'].value_counts().nlargest(5).index
        df_top_bodies = df_sample[df_sample['body'].isin(top_bodies)]
        fig10 = px.histogram(df_top_bodies, x='state', color='body', barmode='group', title="Top 5 Tipos de Carrocería por Estado")
        st.plotly_chart(fig10)

    with st.expander("Odómetro vs Precio por Estado"):
        fig11 = px.scatter(df_sample, x='odometer', y='sellingprice', color='state', opacity=0.4)
        st.plotly_chart(fig11)

    with st.expander("Correlación entre Estado, MMR y Precio"):
        @st.cache_data
        def modelo_estado(df):
            X = df[['mmr']]
            y = df['sellingprice']
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            return X['mmr'], y, y_pred, r2, mse

        x_mmr, y_real, y_pred, r2, mse = modelo_estado(df_sample)
        fig7 = px.scatter(df_sample, x=x_mmr, y=y_real, color='state', title="Relación MMR vs Precio de Venta por Estado")
        fig7.add_traces(go.Scatter(x=x_mmr, y=y_pred, mode='lines', name='Regresión lineal', line=dict(color='red')))
        st.plotly_chart(fig7)
        st.markdown(f"**R²:** {r2:.4f} | **MSE:** {mse:,.2f}")

    with st.expander("Distribución Total de Precios por Estado"):
        state_totals = df_clean.groupby('state')['sellingprice'].sum().reset_index()
        top_states = state_totals.sort_values(by='sellingprice', ascending=False).head(10)
        fig6 = px.pie(top_states, names='state', values='sellingprice', title="Top 10 Estados por Total de Precio de Venta")
        fig6.update_traces(textinfo='label+percent', textposition='outside')
        st.plotly_chart(fig6)

    with st.expander("Distribución de Autos por Marca y Estado"):
        state_make_counts = df_clean.groupby(['state', 'make']).size().reset_index(name='count')
        fig_treemap = px.treemap(state_make_counts, path=['state', 'make'], values='count', title="Distribución de Autos por Marca y Estado", color='count', color_continuous_scale='blues')
        st.plotly_chart(fig_treemap)

# Créditos del equipo al final del dashboard
st.markdown("""
---
<div style='text-align: center;'>
<h4>Equipo de Análisis de Datos de AutoMarket USA</h4>
<p>Erick Yahir Meza Hernández – Analista de Datos<br>
Julissa Gutiérrez Figueroa – Científica de Datos<br>
David Alfonso Lorenzo – Analista de Datos<br>
Sofie Grajales Gosvig – Científica de Datos</p>
</div>
""", unsafe_allow_html=True)