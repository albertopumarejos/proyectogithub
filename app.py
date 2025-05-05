import streamlit as st
from influxdb_client import InfluxDBClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración desde archivo local
from config import INFLUX_URL, INFLUX_TOKEN, ORG, BUCKET

def get_data(field):
    query = f'''
    from(bucket: "{BUCKET}")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "airSensor")
      |> filter(fn: (r) => r._field == "{field}")
    '''
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=ORG)
    df = client.query_api().query_data_frame(org=ORG, query=query)
    df = df[["_time", "_value"]].rename(columns={"_time": "timestamp", "_value": field})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def detectar_anomalias(df, variable):
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(df[[variable]])
    return df

# --- Streamlit UI ---
st.title("Análisis de Temperatura y Humedad con IA Local")

# Botón para temperatura
if st.button("Cargar y analizar datos de TEMPERATURA"):
    variable = "temperature"
    df = get_data(variable)

    st.subheader("Datos crudos:")
    st.dataframe(df)

    st.subheader("Estadísticas descriptivas:")
    st.write(df[variable].describe())

    df = detectar_anomalias(df, variable)
    outliers = df[df["anomaly"] == -1]

    st.subheader("Visualización con anomalías:")
    fig, ax = plt.subplots()
    sns.lineplot(x="timestamp", y=variable, data=df, label=variable.capitalize(), ax=ax)
    ax.scatter(outliers["timestamp"], outliers[variable], color="red", label="Anomalía", zorder=5)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Anomalías detectadas:")
    st.dataframe(outliers)

# Botón para humedad
if st.button("Cargar y analizar datos de HUMEDAD"):
    variable = "humidity"
    df = get_data(variable)

    st.subheader("Datos crudos:")
    st.dataframe(df)

    st.subheader("Estadísticas descriptivas:")
    st.write(df[variable].describe())

    df = detectar_anomalias(df, variable)
    outliers = df[df["anomaly"] == -1]

    st.subheader("Visualización con anomalías:")
    fig, ax = plt.subplots()
    sns.lineplot(x="timestamp", y=variable, data=df, label=variable.capitalize(), ax=ax)
    ax.scatter(outliers["timestamp"], outliers[variable], color="red", label="Anomalía", zorder=5)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Anomalías detectadas:")
    st.dataframe(outliers)

