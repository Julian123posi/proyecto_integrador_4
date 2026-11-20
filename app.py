import duckdb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configuración general de la página Streamlit
st.set_page_config(
    page_title="Optimización Inferencial Agrícola",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# FASE 1: INGESTA DETERMINISTA (DUCKDB)
# ==========================================
@st.cache_data
def cargar_datos():
    con = duckdb.connect(':memory:')
    con.execute("SELECT setseed(0.42);")
    query_crear_tabla = """
    CREATE TABLE parcelas AS
    WITH base_datos AS (
        SELECT 
            range AS parcela_id,
            ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,
            ROUND(50.0 + random() * 100.0, 2) AS fertilizante_kg,
            CASE 
                WHEN random() < 0.50 THEN 'Tratamiento_A' 
                ELSE 'Tratamiento_B' 
            END AS tipo_tratamiento,
            (-0.8 + random() * 1.6) AS ruido
        FROM range(1, 101)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        ROUND(
            5.0 
            + (0.035 * volumen_riego) 
            + (0.020 * fertilizante_kg) 
            + (CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END) 
            + ruido, 
        2) AS rendimiento_ton
    FROM base_datos;
    """
    con.execute(query_crear_tabla)
    return con.execute("SELECT * FROM parcelas;").df()

df_completo = cargar_datos()

# ==========================================
# FASE 5 (PARTE A): CONTROLES INTERACTIVOS
# ==========================================
st.sidebar.header("🎛️ Filtros de Telemetría")

opciones_tratamiento = st.sidebar.multiselect(
    "Selecciona el Tipo de Tratamiento:",
    options=["Tratamiento_A", "Tratamiento_B"],
    default=["Tratamiento_A", "Tratamiento_B"]
)

rango_riego = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    min_value=float(df_completo["volumen_riego"].min()),
    max_value=float(df_completo["volumen_riego"].max()),
    value=(float(df_completo["volumen_riego"].min()), float(df_completo["volumen_riego"].max()))
)

# Aplicar filtros al DataFrame
df_filtrado = df_completo[
    (df_completo["tipo_tratamiento"].isin(opciones_tratamiento)) &
    (df_completo["volumen_riego"].between(rango_riego[0], rango_riego[1]))
]

if df_filtrado.empty:
    st.warning("⚠️ No hay parcelas que coincidan con los filtros seleccionados. Ajusta el panel lateral.")
    st.stop()

# ==========================================
# FASE 2: MODELO PREDICTIVO OLS (SOBRE FILTRO)
# ==========================================
Y = df_filtrado['rendimiento_ton']
X = df_filtrado[['volumen_riego', 'fertilizante_kg']]
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

r2_ajustado = modelo_ols.rsquared_adj
predicciones = modelo_ols.predict(X_const)
residuos = Y - predicciones
rmse = np.sqrt(np.mean(residuos ** 2))

# ==========================================
# FASE 3: PRUEBA INFERENCIAL t-STUDENT (GLOBAL)
# ==========================================
# Se calcula sobre el dataset completo para mantener la validez de la comparación A vs B
grupo_A = df_completo[df_completo['tipo_tratamiento'] == 'Tratamiento_A']['rendimiento_ton']
grupo_B = df_completo[df_completo['tipo_tratamiento'] == 'Tratamiento_B']['rendimiento_ton']
t_stat, p_valor_t = stats.ttest_ind(grupo_B, grupo_A, equal_var=False, alternative='greater')
res_bilateral = stats.ttest_ind(grupo_B, grupo_A, equal_var=False, alternative='two-sided')
ic_95 = res_bilateral.confidence_interval(confidence_level=0.95)

# ==========================================
# FASE 5 (PARTE B): ENCABEZADO Y TARJETAS KPI
# ==========================================
st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.markdown("Monitor de telemetría agroindustrial ($N = 100$ parcelas) — Análisis de Regresión OLS e Inferencia de Tratamientos.")

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="Rendimiento Promedio (Selección)",
        value=f"{df_filtrado['rendimiento_ton'].mean():.2f} ton/ha",
        delta=f"{len(df_filtrado)} parcelas activas"
    )

with kpi2:
    st.metric(
        label="R² Ajustado del Modelo OLS",
        value=f"{r2_ajustado * 100:.2f}%",
        delta=f"RMSE: {rmse:.2f} ton/ha",
        delta_color="off"
    )

with kpi3:
    estado_hipotesis = "Significativo (p < 0.05)" if p_valor_t < 0.05 else "No significativo"
    st.metric(
        label="P-Valor Prueba t (B vs A)",
        value=f"{p_valor_t:.2e}",
        delta=estado_hipotesis
    )

st.divider()

# ==========================================
# FASE 4 Y 5 (PARTE C): CUADRÍCULA 2x2 GRÁFICAS
# ==========================================
sns.set_theme(style="whitegrid")

# Fila 1: Gráfica 1 y Gráfica 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Mapa de Calor de Correlación (Pearson)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    corr_matrix = df_filtrado[['volumen_riego', 'fertilizante_kg', 'rendimiento_ton']].corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax1)
    ax1.set_title("Matriz de Correlación de Variables")
    st.pyplot(fig1)

with col2:
    st.subheader("2. Dispersión con Ajuste OLS (Riego vs Rendimiento)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.regplot(
        data=df_filtrado,
        x="volumen_riego",
        y="rendimiento_ton",
        scatter_kws={"alpha": 0.7, "color": "#2b5c8f"},
        line_kws={"color": "#d9534f", "label": "Recta OLS + IC 95%"},
        ax=ax2
    )
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    ax2.set_title("Regresión Lineal: Riego vs. Cosecha")
    ax2.legend()
    st.pyplot(fig2)

# Fila 2: Gráfica 3 y Gráfica 4
col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Diagrama de Cajas Comparativo (Tratamientos)")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=df_filtrado,
        x="tipo_tratamiento",
        y="rendimiento_ton",
        palette=["#8da0cb", "#66c2a5"],
        hue="tipo_tratamiento",
        legend=False,
        ax=ax3
    )
    ax3.set_xlabel("Tipo de Tratamiento")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    ax3.set_title("Distribución de Rendimiento: A (Estándar) vs B (Bio-estimulante)")
    st.pyplot(fig3)

with col4:
    st.subheader("4. Histograma y Densidad de Residuos")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.histplot(residuos, kde=True, color="#4daf4a", bins=15, ax=ax4)
    ax4.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Media Cero (0.0)")
    ax4.set_xlabel("Residuos (Y real - Y predicho)")
    ax4.set_ylabel("Frecuencia")
    ax4.set_title("Diagnóstico de Normalidad de Errores")
    ax4.legend()
    st.pyplot(fig4)

# ==========================================
# SECCIÓN EXTRA: VEREDICTO Y TABLA DE DATOS
# ==========================================
st.divider()
st.subheader("📋 Veredicto Técnico de Ingeniería")
st.info(
    f"**Resultado de la Prueba $t$-Student:** La media del **Tratamiento B** es **{grupo_B.mean():.2f} ton/ha** "
    f"frente a **{grupo_A.mean():.2f} ton/ha** del **Tratamiento A** (diferencia de **+{grupo_B.mean() - grupo_A.mean():.2f} ton/ha**, "
    f"IC 95%: `[{ic_95.low:.2f}, {ic_95.high:.2f}]`). Con un $p$-valor de **{p_valor_t:.2e} < 0.05**, "
    f"se rechaza $H_0$ y **se recomienda autorizar la inversión para escalar el Tratamiento B**."
)

with st.expander("Ver tabla de datos filtrados (DuckDB -> Pandas)"):
    st.dataframe(df_filtrado, use_container_width=True)