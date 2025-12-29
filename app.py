import streamlit as st
import plotly.express as px
import pandas as pd
import asyncio
import time

from analysis import (
    load_data,
    seasonal_statistics,
    stl_decomposition
)
from parallel_analysis import run_parallel_analysis
from weather_api import (
    get_current_temperature_sync,
    get_current_temperature_async
)

st.set_page_config(page_title="Temperature Analysis", layout="wide")
st.title("🌍 Анализ температур и мониторинг погоды")


# ---------- CACHE ----------

@st.cache_data
def cached_analysis(df):
    return run_parallel_analysis(df)


@st.cache_data
def cached_seasonal_stats(df):
    return seasonal_statistics(df)


# ---------- UI ----------

uploaded_file = st.file_uploader(
    "Загрузите файл temperature_data.csv",
    type="csv"
)

api_key = st.text_input(
    "Введите OpenWeatherMap API Key",
    type="password"
)

api_mode = st.radio(
    "Способ получения температуры",
    ["Синхронный", "Асинхронный"]
)


# ---------- DATA ----------

if uploaded_file:
    df = load_data(uploaded_file)

    with st.spinner("Выполняется анализ (с кешированием)..."):
        df = cached_analysis(df)

    stats = cached_seasonal_stats(df)

    city = st.selectbox(
        "Выберите город",
        sorted(df["city"].unique())
    )

    df_city = df[df["city"] == city]


    # ---------- DESCRIPTIVE ----------

    st.subheader("📊 Описательная статистика")
    st.dataframe(df_city["temperature"].describe())


    # ---------- TIME SERIES ----------

    st.subheader("📈 Температурный ряд и аномалии")

    fig_ts = px.line(
        df_city,
        x="timestamp",
        y="temperature"
    )

    fig_ts.add_scatter(
        x=df_city[df_city["anomaly"]]["timestamp"],
        y=df_city[df_city["anomaly"]]["temperature"],
        mode="markers",
        name="Аномалии"
    )

    st.plotly_chart(fig_ts, use_container_width=True)


    # ---------- STL ----------

    st.subheader("🔍 STL-декомпозиция временного ряда")

    stl_df = stl_decomposition(df_city)

    fig_trend = px.line(stl_df, x="timestamp", y="trend", title="Тренд")
    fig_seasonal = px.line(stl_df, x="timestamp", y="seasonal", title="Сезонность")
    fig_resid = px.line(stl_df, x="timestamp", y="resid", title="Остатки")

    st.plotly_chart(fig_trend, use_container_width=True)
    st.plotly_chart(fig_seasonal, use_container_width=True)
    st.plotly_chart(fig_resid, use_container_width=True)


    # ---------- SEASONAL PROFILES ----------

    st.subheader("🌦 Сезонные профили")

    fig_season = px.bar(
        stats[stats["city"] == city],
        x="season",
        y="mean",
        error_y="std"
    )

    st.plotly_chart(fig_season, use_container_width=True)


    # ---------- CURRENT TEMPERATURE ----------

    st.subheader("🌡 Текущая температура")

    if api_key:
        try:
            start = time.time()

            if api_mode == "Синхронный":
                current_temp = get_current_temperature_sync(city, api_key)
            else:
                current_temp = asyncio.run(
                    get_current_temperature_async(city, api_key)
                )

            elapsed = time.time() - start

            current_season = df_city.sort_values("timestamp")["season"].iloc[-1]

            row = stats[
                (stats["city"] == city) &
                (stats["season"] == current_season)
            ].iloc[0]

            is_normal = (
                row["mean"] - 2 * row["std"]
                <= current_temp
                <= row["mean"] + 2 * row["std"]
            )

            st.metric(
                f"Температура сейчас в {city}",
                f"{current_temp:.1f} °C",
                "Норма" if is_normal else "Аномалия"
            )

            st.caption(
                f"⏱ Время запроса ({api_mode.lower()}): {elapsed:.3f} сек"
            )

        except ValueError as e:
            st.error(e)

    else:
        st.info("Введите API Key для получения текущей температуры")
