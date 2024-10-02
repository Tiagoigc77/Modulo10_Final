import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Título de la aplicación
st.title("Proyecto Final: Clustering con K-Means y Clustering Jerárquico con Streamlit")
st.write("Integrantes del grupo: Jose Mario Dorado Sorich, Nelson Estrada Alcoba y Santiago Ignacio Gallardo Castro")

# Subir archivo CSV localmente
uploaded_file = st.file_uploader("Sube tu archivo CSV para clustering", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file, sep=';')  # Cambia el delimitador si es necesario

    # Mostrar los primeros datos cargados
    st.write("Datos cargados:")
    st.dataframe(data.head())

    # Mostrar las columnas con valores nulos
    st.write("Columnas con valores nulos:")
    null_columns = data.columns[data.isnull().any()].tolist()
    if null_columns:
        st.write(f"Columnas que contienen valores nulos: {null_columns}")
        # Eliminar las columnas con valores nulos
        data = data.drop(columns=null_columns)
        st.write("Columnas con valores nulos eliminadas.")
    else:
        st.write("No hay columnas con valores nulos.")

    # Estadísticas descriptivas
    st.write("Estadísticas descriptivas:")
    st.write(data.describe())

    # Mostrar gráficos para las variables numéricas
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    st.write("Gráficos de las variables numéricas:")
    for col in numerical_columns:
        fig, ax = plt.subplots()
        ax.hist(data[col].dropna(), bins=30, color='skyblue')
        ax.set_title(f'Histograma de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig)

    # Menús de selección en la barra lateral
    st.sidebar.title("Seleccione el modelo de clustering")
    model_type = st.sidebar.radio(
        "Elige el tipo de clustering que deseas visualizar",
        ('KMeans', 'Clustering Jerárquico')
    )

    # Seleccionar columnas numéricas para clustering
    selected_columns = st.sidebar.multiselect("Selecciona las columnas para clustering", numerical_columns)

    if selected_columns:
        # Preprocesar los datos: Escalar
        X = data[selected_columns].dropna()  # Eliminar filas con valores NaN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == 'KMeans':
            # 1. Seleccionar el número de clústeres (K)
            n_clusters = st.sidebar.slider("Selecciona el número de clústeres (K)", 2, 10, 3)

            # 2. Aplicar K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)
            data['Cluster'] = kmeans.labels_

            # 3. Mostrar los resultados
            st.write(f"Clustering completado con {n_clusters} clústeres.")
            st.dataframe(data)

            # 4. Visualización (gráfico de dispersión usando las primeras dos columnas)
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
                ax.set_xlabel(selected_columns[0])
                ax.set_ylabel(selected_columns[1])
                ax.set_title(f'Clustering K-Means con {n_clusters} clústeres')
                st.pyplot(fig)
            else:
                st.warning("Selecciona al menos 2 columnas numéricas para la visualización.")

        elif model_type == 'Clustering Jerárquico':
            # 1. Seleccionar el número de clústeres para el clustering jerárquico
            n_clusters = st.sidebar.slider("Selecciona el número de clústeres para clustering jerárquico", 2, 10, 3)

            # 2. Aplicar Clustering Jerárquico
            hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = hierarchical_clustering.fit_predict(X_scaled)
            data['Cluster'] = labels

            # 3. Mostrar los resultados
            st.write(f"Clustering jerárquico completado con {n_clusters} clústeres.")
            st.dataframe(data)

            # 4. Visualización del dendrograma
            st.write("Dendrograma del Clustering Jerárquico:")
            linked = linkage(X_scaled, 'ward')  # Usamos el método de Ward para la fusión
            fig, ax = plt.subplots(figsize=(10, 7))  # Ajustar el tamaño del gráfico
            dendrogram(linked, ax=ax, truncate_mode='lastp', p=n_clusters)
            ax.set_title(f'Dendrograma con {n_clusters} clústeres')
            st.pyplot(fig)

    else:
        st.warning("Selecciona al menos una columna numérica para el clustering.")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")


