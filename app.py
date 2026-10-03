"""
Лабораторная работа № 3.1 — веб-приложение на Streamlit
Сквозной проект, вариант № 8: Мониторинг и оценка эффективности работы отелей
Студент: Момунов Акыл, ПИ-2-23
Запуск:  streamlit run app.py   (hotel_clean.csv должен лежать рядом)
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Мониторинг отелей", page_icon="🏨", layout="wide")
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ данные
@st.cache_data
def load(src):
    df = pd.read_csv(src, parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df

try:
    df = load("hotel_clean.csv")
except FileNotFoundError:
    up = st.sidebar.file_uploader("Загрузите hotel_clean.csv", type="csv")
    if up is None:
        st.info("Положите hotel_clean.csv рядом с app.py или загрузите файл в боковой панели.")
        st.stop()
    df = load(up)

# ------------------------------------------------------------------ модели (кэшируются)
REG_FEATURES = {
    "stars": "Звёздность", "rooms_total": "Номерной фонд", "occupancy_rate": "Загрузка (0–1)",
    "avg_rating": "Рейтинг гостей", "competitor_adr_som": "Чек конкурентов, сом",
    "marketing_spend_som": "Маркетинг, сом/день", "is_weekend": "Выходной", "is_holiday": "Праздник",
}
PROBIT_FEATURES = ["avg_rating", "adr_k", "marketing_k", "temperature_c", "is_weekend", "is_holiday", "stars"]
CL_FEATURES = ["occupancy_rate", "adr_som", "avg_rating", "avg_stay_nights", "cancel_rate", "marketing_spend_som"]

@st.cache_resource
def train_regression(data):
    X, y = data[list(REG_FEATURES)], data["adr_som"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    ols = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
    rf = RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    res = {}
    p_ols = ols.predict(sm.add_constant(Xte, has_constant="add"))
    p_rf = rf.predict(Xte)
    for name, p in [("Линейная регрессия (МНК)", p_ols), ("Random Forest", p_rf)]:
        res[name] = (r2_score(yte, p), np.sqrt(mean_squared_error(yte, p)))
    return ols, rf, res

@st.cache_resource
def train_probit(data):
    d = data.assign(adr_k=data["adr_som"] / 1000, marketing_k=data["marketing_spend_som"] / 1000)
    y = (d["occupancy_rate"] >= 0.70).astype(int)          # 1 = высокая загрузка (≥ 70 %)
    X = sm.add_constant(d[PROBIT_FEATURES])
    model = sm.Probit(y, X).fit(disp=0)
    return model, X.mean(), y.mean()

ols, rf, reg_scores = train_regression(df)
probit, x_mean, share_high = train_probit(df)

# ------------------------------------------------------------------ шапка и фильтры
st.title("🏨 Мониторинг и оценка эффективности работы отелей")
st.caption("Сквозной проект по дисциплине «Big Data и анализ данных» · Момунов Акыл, ПИ-2-23 · вариант № 8")

st.sidebar.header("Фильтры")
cities = st.sidebar.multiselect("Город", sorted(df["city"].unique()), default=sorted(df["city"].unique()))
hotels_av = sorted(df[df["city"].isin(cities)]["hotel_name"].unique())
hotels = st.sidebar.multiselect("Отели", hotels_av, default=hotels_av)
dmin, dmax = df["date"].min().date(), df["date"].max().date()
period = st.sidebar.date_input("Период", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(period) != 2:
    st.stop()
f = df[df["hotel_name"].isin(hotels) & df["date"].between(pd.Timestamp(period[0]), pd.Timestamp(period[1]))]
if f.empty:
    st.warning("По выбранным фильтрам данных нет.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Обзор", "🔎 Исследование данных", "🧩 Кластеры отелей", "💰 Прогноз среднего чека", "📈 Вероятность высокой загрузки"])

# ------------------------------------------------------------------ 1. Обзор
with tab1:
    c = st.columns(5)
    c[0].metric("Средняя загрузка", f"{f['occupancy_rate'].mean():.1%}")
    c[1].metric("Средний чек (ADR)", f"{f['adr_som'].mean():,.0f} сом")
    c[2].metric("RevPAR", f"{f['revpar_som'].mean():,.0f} сом")
    c[3].metric("Средний рейтинг", f"{f['avg_rating'].mean():.2f}")
    c[4].metric("Выручка за период", f"{f['revenue_som'].sum() / 1e6:,.1f} млн")

    left, right = st.columns(2)
    with left:
        st.subheader("Загрузка номеров по месяцам")
        st.line_chart(f.pivot_table(index="month", columns="hotel_name", values="occupancy_rate"))
    with right:
        st.subheader("Рейтинг отелей по RevPAR")
        top = f.groupby("hotel_name")["revpar_som"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        top.plot.barh(ax=ax, color="#3b82f6"); ax.set_xlabel("RevPAR, сом")
        st.pyplot(fig)
    st.subheader("Сезонность")
    order = ["Зима", "Весна", "Лето", "Осень"]
    seas = f.groupby(["season", "city"])["occupancy_rate"].mean().unstack().reindex(order)
    st.bar_chart(seas)

# ------------------------------------------------------------------ 2. EDA
with tab2:
    col = st.selectbox("Показатель для гистограммы", ["occupancy_rate", "adr_som", "avg_rating", "revpar_som", "cancel_rate", "avg_stay_nights"])
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4)); sns.histplot(f[col], bins=40, kde=True, ax=ax)
        ax.set_title(f"Распределение: {col}"); st.pyplot(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4)); sns.boxplot(data=f, x="stars", y=col, ax=ax)
        ax.set_title("По звёздности отеля"); st.pyplot(fig)

    a, b = st.columns(2)
    with a:
        st.subheader("Корреляционная матрица")
        cols = ["occupancy_rate", "adr_som", "avg_rating", "avg_stay_nights", "cancel_rate", "marketing_spend_som", "competitor_adr_som", "temperature_c"]
        fig, ax = plt.subplots(figsize=(6.5, 5)); sns.heatmap(f[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    with b:
        st.subheader("Автокорреляция загрузки")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        autocorrelation_plot(f.groupby("date")["occupancy_rate"].mean(), ax=ax)
        st.pyplot(fig)
        st.caption("Высокие значения на малых лагах — загрузка сохраняет «инерцию» от дня к дню.")

# ------------------------------------------------------------------ 3. Кластеры
with tab3:
    st.write("Кластеризация K-means по показателям эффективности (данные стандартизированы).")
    X = StandardScaler().fit_transform(df[CL_FEATURES])
    a, b = st.columns(2)
    with a:
        st.subheader("Метод локтя")
        ks = range(2, 9)
        inertia = [KMeans(k, n_init=5, random_state=42).fit(X).inertia_ for k in ks]
        fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(list(ks), inertia, "o-"); ax.set_xlabel("k"); ax.set_ylabel("Inertia")
        st.pyplot(fig)
    with b:
        k = st.slider("Число кластеров k", 2, 8, 4)
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        dc = df.assign(cluster=km.labels_)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=dc, x="occupancy_rate", y="adr_som", hue="cluster", palette="tab10", alpha=.5, s=12, ax=ax)
        st.pyplot(fig)
    st.subheader("Средние профили кластеров")
    prof = dc.groupby("cluster")[CL_FEATURES].mean().round(2)
    prof["наблюдений"] = dc.groupby("cluster").size()
    st.dataframe(prof)
    st.subheader("Состав кластеров по отелям")
    st.dataframe(pd.crosstab(dc["hotel_name"], dc["cluster"]))

# ------------------------------------------------------------------ 4. Прогноз ADR
with tab4:
    st.write("Прогноз среднего чека (ADR) по параметрам отеля и дня.")
    m1, m2 = st.columns(2)
    for col_, (name, (r2, rmse)) in zip((m1, m2), reg_scores.items()):
        col_.metric(name, f"R² = {r2:.3f}", f"RMSE = {rmse:,.0f} сом", delta_color="off")

    model_name = st.radio("Модель", list(reg_scores), horizontal=True)
    s1, s2 = st.columns(2)
    inp = {}
    with s1:
        inp["stars"] = st.select_slider("Звёздность", [2, 3, 4, 5], 4)
        inp["rooms_total"] = st.slider("Номерной фонд", 30, 150, 80)
        inp["occupancy_rate"] = st.slider("Загрузка номеров", 0.10, 1.0, 0.65, 0.01)
        inp["avg_rating"] = st.slider("Рейтинг гостей", 2.5, 5.0, 4.2, 0.1)
    with s2:
        inp["competitor_adr_som"] = st.slider("Чек конкурентов, сом", 1500, 15000, 6000, 100)
        inp["marketing_spend_som"] = st.slider("Маркетинг, сом/день", 500, 5000, 2000, 50)
        inp["is_weekend"] = int(st.checkbox("Выходной день"))
        inp["is_holiday"] = int(st.checkbox("Праздничный день"))
    row = pd.DataFrame([inp])[list(REG_FEATURES)]
    pred = (ols.predict(sm.add_constant(row, has_constant="add")).iloc[0]
            if model_name.startswith("Лин") else rf.predict(row)[0])
    st.success(f"Прогнозируемый средний чек: **{pred:,.0f} сом**")
    with st.expander("Коэффициенты линейной модели (МНК)"):
        t = pd.DataFrame({"коэф.": ols.params, "t-стат.": ols.tvalues, "p-value": ols.pvalues}).round(4)
        st.dataframe(t); st.caption(f"R² = {ols.rsquared:.3f}, F-критерий = {ols.fvalue:,.1f}")

# ------------------------------------------------------------------ 5. Probit
with tab5:
    st.write(f"Probit-модель: вероятность того, что загрузка отеля **≥ 70 %** (доля таких дней в данных: {share_high:.0%}).")
    a, b = st.columns(2)
    with a:
        p_rating = st.slider("Рейтинг", 2.5, 5.0, 4.0, 0.1, key="p1")
        p_adr = st.slider("Средний чек, сом", 1500, 15000, 6000, 100, key="p2")
        p_mkt = st.slider("Маркетинг, сом/день", 500, 5000, 2000, 50, key="p3")
    with b:
        p_temp = st.slider("Температура, °C", -20, 35, 20, key="p4")
        p_stars = st.select_slider("Звёздность", [2, 3, 4, 5], 4, key="p5")
        p_we = int(st.checkbox("Выходной", key="p6")); p_h = int(st.checkbox("Праздник", key="p7"))
    xi = pd.DataFrame([[1, p_rating, p_adr / 1000, p_mkt / 1000, p_temp, p_we, p_h, p_stars]],
                      columns=["const"] + PROBIT_FEATURES)
    prob = float(probit.predict(xi).iloc[0])
    st.metric("Вероятность высокой загрузки", f"{prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.subheader("Предельные эффекты (для «среднего» отеля)")
    me = probit.get_margeff(at="mean").summary_frame()
    st.dataframe(me[["dy/dx", "Pr(>|z|)"]].round(4).rename(columns={"dy/dx": "Δ вероятности", "Pr(>|z|)": "p-value"}))
    z = float(probit.params @ x_mean)
    st.caption(f"Вероятность для среднего объекта выборки: Φ({z:.2f}) = {norm.cdf(z):.1%}")
