# 🇨🇴 SpVAR Colombia – Streamlit App

Aplicación **Streamlit** del modelo Vector Autorregresivo Espacial (SpVAR) para Educación y Crecimiento Económico en los departamentos de Colombia.

Compatible con **Streamlit Cloud**.

## Características

- Mapas coropléticos de alto espectro
- Mapa interactivo Folium (CartoDB dark_matter)
- Visualizaciones 3D (Scatter, Superficie, Animación)
- 4 mecanismos de transmisión espacial
- Impactos LeSage & Pace
- Informe ejecutivo descargable

## Fuentes 100% públicas (sin API keys)

- GeoJSON: https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson
- Tiles: CartoDB dark_matter
- Datos: sintéticos generados localmente

## Despliegue en Streamlit Cloud

1. Sube esta carpeta a un repositorio de GitHub
2. En [share.streamlit.io](https://share.streamlit.io) → New app
3. Selecciona el repositorio y el archivo `app.py`
4. Deploy

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```
spvar_colombia_streamlit/
├── app.py
├── requirements.txt
└── README.md
```
