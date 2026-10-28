"""
SpVAR Colombia – Educación y Crecimiento Económico
Versión Streamlit (compatible con Streamlit Cloud)
Fuentes 100% públicas · Sin API keys
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran
from statsmodels.regression.linear_model import OLS as sm_OLS
from statsmodels.tools.tools import add_constant
from scipy.linalg import inv
from scipy.interpolate import griddata

# -----------------------------------------------------------------------------
# Configuración de página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpVAR Colombia | Educación & Crecimiento",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tema oscuro suave vía CSS
st.markdown(
    """
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    h1, h2, h3, p, label, .stMarkdown { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    div[data-testid="stMetricValue"] { color: #22d3ee; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Carga y procesamiento de datos (cacheado para Streamlit Cloud)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando geometría y datos espaciales...")
def load_and_process_data():
    URL_GEOJSON = (
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/"
        "co_2018_MGN_DPTO_POLITICO.geojson"
    )
    gdf = gpd.read_file(URL_GEOJSON)
    gdf = gdf.rename(columns={"DPTO_CCDGO": "cod_dpto", "DPTO_CNMBR": "departamento"})
    gdf["cod_dpto"] = gdf["cod_dpto"].astype(str).str.zfill(2)
    gdf = gdf.to_crs("EPSG:4326")

    np.random.seed(42)
    N = len(gdf)
    T = 10
    years = list(range(2015, 2015 + T))

    w_true = Queen.from_dataframe(gdf, use_index=False)
    w_true.transform = "r"

    educ = np.zeros((T, N))
    growth = np.zeros((T, N))
    viirs = np.zeros((T, N))

    educ[0] = 8.2 + 1.8 * np.random.randn(N)
    growth[0] = 2.8 + 1.4 * np.random.randn(N)
    viirs[0] = 12 + 4.5 * np.random.randn(N)

    for t in range(1, T):
        educ[t] = (
            0.62 * educ[t - 1]
            + 0.19 * lag_spatial(w_true, educ[t - 1])
            + 0.09 * growth[t - 1]
            + 0.35 * np.random.randn(N)
        )
        growth[t] = (
            0.51 * growth[t - 1]
            + 0.21 * lag_spatial(w_true, growth[t - 1])
            + 0.28 * educ[t - 1]
            + 0.14 * lag_spatial(w_true, educ[t - 1])
            + 0.85 * np.random.randn(N)
        )
        viirs[t] = (
            0.70 * viirs[t - 1]
            + 0.15 * lag_spatial(w_true, viirs[t - 1])
            + 0.40 * growth[t]
            + 1.1 * np.random.randn(N)
        )

    df_long = pd.DataFrame(
        {
            "year": np.repeat(years, N),
            "departamento": np.tile(gdf["departamento"].values, T),
            "cod_dpto": np.tile(gdf["cod_dpto"].values, T),
            "educacion": educ.flatten(),
            "crecimiento": growth.flatten(),
            "viirs": viirs.flatten(),
        }
    )

    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y
    df_long = df_long.merge(
        gdf[["departamento", "lon", "lat"]], on="departamento", how="left"
    )

    min_growth = df_long["crecimiento"].min()
    df_long["crecimiento_pos"] = df_long["crecimiento"] - min_growth + 1.5

    # Selección de W
    weights = {
        "Queen": Queen.from_dataframe(gdf, use_index=False),
        "Rook": Rook.from_dataframe(gdf, use_index=False),
    }
    for k in [3, 4, 5]:
        weights[f"KNN_{k}"] = KNN.from_dataframe(gdf, k=k, use_index=False)
    for name, w in weights.items():
        w.transform = "r"

    avg_educ = (
        df_long.groupby("departamento")["educacion"]
        .mean()
        .reindex(gdf["departamento"])
        .values
    )
    gdf["avg_educ"] = avg_educ

    best_name, best_W = "Queen", weights["Queen"]
    best_abs_i = -1
    for name, w in weights.items():
        mi = Moran(gdf["avg_educ"].values, w)
        if mi.p_sim < 0.05 and abs(mi.I) > best_abs_i:
            best_abs_i = abs(mi.I)
            best_name, best_W = name, w

    avg_data = (
        df_long.groupby("departamento")
        .agg({"educacion": "mean", "crecimiento": "mean", "viirs": "mean"})
        .reset_index()
    )
    gdf_map = gdf.merge(avg_data, on="departamento", how="left")

    # SpVAR
    df = df_long.sort_values(["departamento", "year"]).copy()
    df["educ_l1"] = df.groupby("departamento")["educacion"].shift(1)
    df["growth_l1"] = df.groupby("departamento")["crecimiento"].shift(1)
    df = df.dropna(subset=["educ_l1", "growth_l1"]).reset_index(drop=True)

    rows = []
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr].set_index("departamento").reindex(gdf["departamento"])
        educ_l1 = sub["educ_l1"].values.astype(float)
        growth_l1 = sub["growth_l1"].values.astype(float)
        viirs_val = sub["viirs"].values.astype(float)
        W_educ = lag_spatial(best_W, educ_l1)
        W_growth = lag_spatial(best_W, growth_l1)
        for i, dep in enumerate(gdf["departamento"]):
            rows.append(
                {
                    "year": yr,
                    "departamento": dep,
                    "educacion": float(sub.loc[dep, "educacion"]),
                    "crecimiento": float(sub.loc[dep, "crecimiento"]),
                    "educ_l1": educ_l1[i],
                    "growth_l1": growth_l1[i],
                    "W_educ_l1": W_educ[i],
                    "W_growth_l1": W_growth[i],
                    "viirs": viirs_val[i],
                }
            )
    spvar_df = pd.DataFrame(rows)

    def estimate(y, X, names):
        Xc = add_constant(X)
        m = sm_OLS(y, Xc).fit(cov_type="HC1")
        return dict(zip(["const"] + names, m.params))

    coef_educ = estimate(
        spvar_df["educacion"].values,
        spvar_df[["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"]].values,
        ["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"],
    )
    coef_growth = estimate(
        spvar_df["crecimiento"].values,
        spvar_df[["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"]].values,
        ["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"],
    )

    def impacts(rho, beta, w):
        n = w.n
        S = inv(np.eye(n) - rho * w.full()[0])
        direct = np.mean(np.diag(S)) * beta
        total = np.mean(S.sum(axis=1)) * beta
        return {"direct": direct, "indirect": total - direct, "total": total}

    imp_e = impacts(coef_educ.get("W_educ_l1", 0.0), coef_educ["educ_l1"], best_W)
    imp_g = impacts(coef_growth.get("W_growth_l1", 0.0), coef_growth["growth_l1"], best_W)

    return {
        "gdf": gdf,
        "gdf_map": gdf_map,
        "df_long": df_long,
        "spvar_df": spvar_df,
        "best_W_name": best_name,
        "coef_educ": coef_educ,
        "coef_growth": coef_growth,
        "imp_e": imp_e,
        "imp_g": imp_g,
    }


# -----------------------------------------------------------------------------
# Cargar datos
# -----------------------------------------------------------------------------
data = load_and_process_data()
gdf_map = data["gdf_map"]
df_long = data["df_long"]
spvar_df = data["spvar_df"]
best_W_name = data["best_W_name"]
coef_educ = data["coef_educ"]
coef_growth = data["coef_growth"]
imp_e = data["imp_e"]
imp_g = data["imp_g"]

# -----------------------------------------------------------------------------
# HERO
# -----------------------------------------------------------------------------
st.title("🇨🇴 SpVAR Colombia")
st.subheader("Educación y Crecimiento Económico a nivel Departamental")
st.markdown(
    """
**Análisis espacial avanzado · 33 regiones · 2015-2024**  
Fuentes 100% públicas · Sin API keys

> Descubre cómo la educación de un departamento impacta el crecimiento de sus vecinos  
> y cómo los spillovers espaciales configuran el desarrollo regional.
"""
)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🗺️ Exploración Espacial",
        "🌌 Visualización 3D",
        "📊 SpVAR & Mecanismos",
        "📄 Informe Ejecutivo",
    ]
)

# ===== TAB 1 =====
with tab1:
    st.markdown("### Distribución geográfica de las variables clave")

    if st.button("Generar mapas coropléticos", type="primary", key="btn_mapas"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        fig.patch.set_facecolor("#0f172a")
        for ax in axes:
            ax.set_facecolor("#0f172a")

        gdf_map.plot(
            column="educacion",
            cmap="YlGnBu",
            linewidth=0.4,
            edgecolor="white",
            legend=True,
            ax=axes[0],
            legend_kwds={"shrink": 0.5},
        )
        axes[0].set_title("Educación promedio", fontweight="bold", color="white")
        axes[0].axis("off")

        gdf_map.plot(
            column="crecimiento",
            cmap="RdYlGn",
            linewidth=0.4,
            edgecolor="white",
            legend=True,
            ax=axes[1],
            legend_kwds={"shrink": 0.5},
        )
        axes[1].set_title("Crecimiento económico", fontweight="bold", color="white")
        axes[1].axis("off")

        gdf_map.plot(
            column="viirs",
            cmap="inferno",
            linewidth=0.4,
            edgecolor="white",
            legend=True,
            ax=axes[2],
            legend_kwds={"shrink": 0.5},
        )
        axes[2].set_title("Luces nocturnas (VIIRS)", fontweight="bold", color="white")
        axes[2].axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### Mapa interactivo multi-capa (CartoDB dark_matter)")
    if st.button("Cargar mapa Folium", key="btn_folium"):
        m = folium.Map(
            location=[4.5, -74.0], zoom_start=5.3, tiles="CartoDB dark_matter"
        )

        def add_layer(gdf, col, name, cmap):
            folium.Choropleth(
                geo_data=gdf.__geo_interface__,
                data=gdf,
                columns=["departamento", col],
                key_on="feature.properties.departamento",
                fill_color=cmap,
                fill_opacity=0.75,
                line_opacity=0.25,
                legend_name=name,
                name=name,
            ).add_to(m)

        add_layer(gdf_map, "viirs", "Luces nocturnas", "YlOrRd")
        add_layer(gdf_map, "educacion", "Educación", "YlGnBu")
        add_layer(gdf_map, "crecimiento", "Crecimiento", "RdYlGn")
        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)
        st_folium(m, width=900, height=500)

# ===== TAB 2 =====
with tab2:
    st.markdown("### Explora las dimensiones espaciales en 3D")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Scatter 3D 2024", type="primary", key="btn_3d"):
            fig = px.scatter_3d(
                df_long[df_long["year"] == 2024],
                x="lon",
                y="lat",
                z="viirs",
                color="educacion",
                size="crecimiento_pos",
                hover_name="departamento",
                color_continuous_scale="Turbo",
                size_max=22,
                title="Educación × Crecimiento × Luces (2024)",
            )
            fig.update_layout(
                template="plotly_dark",
                scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)),
                paper_bgcolor="#0f172a",
                font=dict(color="white"),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if st.button("Superficie de luces", key="btn_surf"):
            gdf_2024 = df_long[df_long["year"] == 2024]
            lon = gdf_2024["lon"].values
            lat = gdf_2024["lat"].values
            z = gdf_2024["viirs"].values
            xi = np.linspace(lon.min(), lon.max(), 70)
            yi = np.linspace(lat.min(), lat.max(), 70)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((lon, lat), z, (Xi, Yi), method="cubic")
            fig = go.Figure(
                data=[go.Surface(x=Xi, y=Yi, z=Zi, colorscale="Inferno")]
            )
            fig.update_layout(
                title="Superficie de intensidad de luces nocturnas",
                template="plotly_dark",
                scene=dict(
                    xaxis_title="Longitud",
                    yaxis_title="Latitud",
                    zaxis_title="VIIRS",
                    aspectmode="manual",
                    aspectratio=dict(x=1.3, y=1, z=0.45),
                ),
                paper_bgcolor="#0f172a",
                font=dict(color="white"),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if st.button("Animación 2015-2024", key="btn_anim"):
            fig = px.scatter_3d(
                df_long,
                x="lon",
                y="lat",
                z="viirs",
                color="educacion",
                size="crecimiento_pos",
                animation_frame="year",
                hover_name="departamento",
                color_continuous_scale="Turbo",
                size_max=18,
                title="Evolución temporal 2015-2024",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                scene=dict(
                    aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)
                ),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3 =====
with tab3:
    st.markdown(
        """
    ### Los cuatro canales de transmisión espacial
    El modelo SpVAR aísla cómo la educación y el crecimiento se transmiten  
    a través del tiempo y del espacio entre los departamentos de Colombia.
    """
    )

    if st.button("Calcular y mostrar mecanismos", type="primary", key="btn_mec"):
        st.markdown(
            f"""
### Los 4 Mecanismos de Transmisión Espacial

**1. Autodinámica espacial** (inercia propia)  
• Educación → φ_ee = **{coef_educ['educ_l1']:.4f}**  
• Crecimiento → φ_gg = **{coef_growth['growth_l1']:.4f}**

**2. Cruce local** (misma región)  
• Educación ← Crecimiento local: **{coef_educ['growth_l1']:.4f}**  
• Crecimiento ← Educación local: **{coef_growth['educ_l1']:.4f}**

**3. Relación espacial directa** (vecinos → misma variable)  
• W·Educación → Educación: **{coef_educ['W_educ_l1']:.4f}**  
• W·Crecimiento → Crecimiento: **{coef_growth['W_growth_l1']:.4f}**

**4. Relación espacial cruzada** (vecinos → otra variable)  
• W·Crecimiento → Educación: **{coef_educ['W_growth_l1']:.4f}**  
• W·Educación → Crecimiento: **{coef_growth['W_educ_l1']:.4f}**

---
### Impactos de largo plazo (LeSage & Pace)

| Variable     | Directo   | Indirecto | Total    |
|--------------|-----------|-----------|----------|
| Educación    | {imp_e['direct']:.4f} | {imp_e['indirect']:.4f} | {imp_e['total']:.4f} |
| Crecimiento  | {imp_g['direct']:.4f} | {imp_g['indirect']:.4f} | {imp_g['total']:.4f} |

**Matriz W seleccionada:** `{best_W_name}`  
**Periodo:** {spvar_df['year'].min()} – {spvar_df['year'].max()}  
**Observaciones:** {len(spvar_df)}
"""
        )

# ===== TAB 4 =====
with tab4:
    st.markdown(
        """
    ### Descarga tu informe personalizado
    Genera un resumen ejecutivo listo para compartir con stakeholders.
    """
    )

    informe = f"""INFORME EJECUTIVO – SpVAR Educación y Crecimiento Económico
Colombia · Departamentos (2015-2024)

Fuente geometría: pública (GitHub raw – caticoa3/colombia_mapa)
Matriz de pesos espaciales: {best_W_name}
Observaciones: {len(spvar_df)}
Periodo: {spvar_df['year'].min()} – {spvar_df['year'].max()}

MECANISMOS DE TRANSMISIÓN
1. Autodinámica espacial
   Educación φ_ee = {coef_educ['educ_l1']:.4f}
   Crecimiento φ_gg = {coef_growth['growth_l1']:.4f}

2. Cruce local
   Educación ← Crecimiento = {coef_educ['growth_l1']:.4f}
   Crecimiento ← Educación = {coef_growth['educ_l1']:.4f}

3. Relación espacial directa
   W·Educ → Educ = {coef_educ['W_educ_l1']:.4f}
   W·Crec → Crec = {coef_growth['W_growth_l1']:.4f}

4. Relación espacial cruzada
   W·Crec → Educ = {coef_educ['W_growth_l1']:.4f}
   W·Educ → Crec = {coef_growth['W_educ_l1']:.4f}

IMPACTOS DE LARGO PLAZO (LeSage & Pace)
Educación  → Directo: {imp_e['direct']:.4f} | Indirecto: {imp_e['indirect']:.4f} | Total: {imp_e['total']:.4f}
Crecimiento → Directo: {imp_g['direct']:.4f} | Indirecto: {imp_g['indirect']:.4f} | Total: {imp_g['total']:.4f}
"""

    st.download_button(
        label="📥 Descargar informe (.txt)",
        data=informe,
        file_name="informe_spvar_colombia.txt",
        mime="text/plain",
        type="primary",
    )

    st.markdown(
        """
---
**Fuente de geometría:** pública (GitHub raw)  
**Datos:** sintéticos realistas con dependencia espacial  
**Sin API keys ni servicios de pago**
"""
    )

# Footer
st.markdown(
    """
<div style="text-align:center; margin-top:2rem; color:#94a3b8; font-size:0.85rem;">
SpVAR Colombia · Modelo Vector Autorregresivo Espacial · Fuentes 100% públicas<br>
Diseñado para experiencia inmersiva y storytelling de datos
</div>
""",
    unsafe_allow_html=True,
)
