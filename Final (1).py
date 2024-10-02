import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Título de la aplicación
st.title("Clustering con K-Means, Clustering Jerárquico y DBSCAN con Streamlit")
st.write("Integrantes del grupo: Nelson Estrada, Jose Dorado y Santiago Gallardo")

# Subir archivo CSV localmente
uploaded_file = st.file_uploader("Sube tu archivo CSV para clustering", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    data = pd.read_csv(uploaded_file, sep=';')  # Cambia el delimitador si es necesario

    # Mostrar los primeros datos cargados
    st.write("Datos cargados:")
    st.dataframe(data.head())

    # Mostrar los tipos de datos de cada columna
    st.write("Tipo de dato de cada columna:")
    st.dataframe(data.dtypes.astype(str))

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

    # Agregar mapa de calor para ver la correlación entre las variables numéricas
    if len(numerical_columns) > 1:
        st.write("Mapa de calor de correlación entre variables numéricas:")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data[numerical_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Menús de selección en la barra lateral
    st.sidebar.title("Seleccione el modelo de clustering")
    model_type = st.sidebar.radio(
        "Elige el tipo de clustering que deseas visualizar",
        ('KMeans', 'Clustering Jerárquico', 'DBSCAN')
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

            # 2. Seleccionar el random_state usando un slider
            random_state = st.sidebar.slider("Selecciona el valor de random_state", 0, 100, 42, step=1)

            # 3. Seleccionar el número de iteraciones
            max_iter = st.sidebar.slider("Selecciona el número máximo de iteraciones", 100, 1000, 300)

            # 4. Aplicar K-Means con los parámetros seleccionados
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
            kmeans.fit(X_scaled)
            data['Cluster'] = kmeans.labels_

            # 5. Mostrar los resultados
            st.write(f"Clustering completado con {n_clusters} clústeres.")
            st.dataframe(data)

            # 6. Visualización (gráfico de dispersión usando las primeras dos columnas)
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
                ax.set_xlabel(selected_columns[0])
                ax.set_ylabel(selected_columns[1])
                ax.set_title(f'Clustering K-Means con {n_clusters} clústeres')

                # Añadir la leyenda con los números de clúster
                legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
                ax.add_artist(legend1)

                st.pyplot(fig)
            else:
                st.warning("Selecciona al menos 2 columnas numéricas para la visualización.")

        elif model_type == 'Clustering Jerárquico':
            # 1. Seleccionar el número de clústeres para el clustering jerárquico
            n_clusters = st.sidebar.slider("Selecciona el número de clústeres para clustering jerárquico", 2, 10, 3)

            # 2. Visualización del dendrograma
            st.write("Dendrograma del Clustering Jerárquico:")
            linked = linkage(X_scaled, 'ward')  # Usamos el método de Ward para la fusión
            fig, ax = plt.subplots(figsize=(10, 7))  # Ajustar el tamaño del gráfico
            dendrogram(linked, ax=ax, truncate_mode='lastp', p=n_clusters, show_leaf_counts=True)
            ax.set_title(f'Dendrograma con {n_clusters} clústeres')
            st.pyplot(fig)

        elif model_type == 'DBSCAN':
            # Parámetros de DBSCAN
            eps = st.sidebar.slider("Selecciona el valor de eps (radio máximo)", 0.1, 10.0, 0.5, step=0.1)
            min_samples = st.sidebar.slider("Selecciona el número mínimo de muestras por vecindario", 1, 10, 5)

            # Aplicar DBSCAN con los parámetros seleccionados
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            data['Cluster'] = labels

            # Mostrar los resultados
            st.write("Clustering DBSCAN completado.")
            st.dataframe(data)

            # Visualización (gráfico de dispersión usando las primeras dos columnas)
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='plasma')
                ax.set_xlabel(selected_columns[0])
                ax.set_ylabel(selected_columns[1])
                ax.set_title(f'Clustering DBSCAN con eps={eps} y min_samples={min_samples}')

                # Añadir la leyenda con los números de clúster
                legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
                ax.add_artist(legend1)

                st.pyplot(fig)
            else:
                st.warning("Selecciona al menos 2 columnas numéricas para la visualización.")

    else:
        st.warning("Selecciona al menos una columna numérica para el clustering.")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")





