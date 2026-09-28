from pathlib import Path
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predicción de retraso de digitación HISMINSA",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "modelo_regresion_digitacion_his.joblib"
CONFIG_PATH = BASE_DIR / "config_modelo_regresion_his.json"
CATALOGO_PATH = BASE_DIR / "catalogo_eess.csv"
HISTORIAL_PATH = BASE_DIR / "historial_eess.csv"


# ============================================================
# CALENDARIO OFICIAL DE CIERRES
# ============================================================
# Se usa para DIAS_HASTA_CIERRE de los meses futuros.
# Para este proyecto se proyecta hasta diciembre de 2026.
# ============================================================

CIERRES_2026 = {
    202601: "2026-02-06",
    202602: "2026-03-04",
    202603: "2026-04-05",
    202604: "2026-05-05",
    202605: "2026-06-04",
    202606: "2026-07-05",
    202607: "2026-08-04",
    202608: "2026-09-04",
    202609: "2026-10-04",
    202610: "2026-11-04",
    202611: "2026-12-04",
    202612: "2027-01-05",
}

PROYECCION_HASTA = pd.Period("2026-12", freq="M")

MESES_ES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>
        /* Baja el contenido para que el botón Deploy no invada el título */
        .block-container {
            padding-top: 4.7rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }

        .titulo-principal {
            font-size: 2.15rem;
            line-height: 1.15;
            font-weight: 800;
            margin: 0 0 .35rem 0;
            color: #1f2937;
        }

        .subtitulo-principal {
            color: #64748b;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .bloque-info {
            border: 1px solid rgba(0,150,186,.34);
            border-left: 6px solid #0096ba;
            background: rgba(0,150,186,.055);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 1rem 0 1.2rem 0;
        }

        .bloque-advertencia {
            border: 1px solid rgba(245,158,11,.35);
            border-left: 6px solid #f59e0b;
            background: rgba(245,158,11,.06);
            border-radius: 12px;
            padding: .9rem 1.1rem;
            margin: .9rem 0 1.2rem 0;
        }

        .credito-sidebar {
            margin-top: 7rem;
            padding-top: 1.1rem;
            border-top: 1px solid rgba(100,116,139,.25);
            font-size: .82rem;
            line-height: 1.45;
            color: #475569;
        }

        .credito-sidebar b {
            color: #1f2937;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_catalogo():
    df = pd.read_csv(CATALOGO_PATH, encoding="utf-8-sig")

    df["ID_ESTABLECIMIENTO"] = pd.to_numeric(
        df["ID_ESTABLECIMIENTO"], errors="coerce"
    ).astype("Int64")

    for col in ["COD_ESTAB", "DESC_ESTAB", "RIS", "CAT_ESTAB", "DESC_DIST"]:
        df[col] = df[col].fillna("SIN DATO").astype(str)

    return df


@st.cache_data
def cargar_historial():
    df = pd.read_csv(HISTORIAL_PATH, encoding="utf-8-sig")

    df["PERIODO"] = pd.to_numeric(
        df["PERIODO"], errors="coerce"
    ).astype("Int64")

    df["ID_ESTABLECIMIENTO"] = pd.to_numeric(
        df["ID_ESTABLECIMIENTO"], errors="coerce"
    ).astype("Int64")

    if "FECHA_CIERRE" in df.columns:
        df["FECHA_CIERRE"] = pd.to_datetime(
            df["FECHA_CIERRE"], errors="coerce"
        )

    return (
        df.sort_values(
            ["ID_ESTABLECIMIENTO", "PERIODO"]
        )
        .reset_index(drop=True)
    )


try:
    modelo = cargar_modelo()
    config = cargar_config()
    catalogo = cargar_catalogo()
    historial = cargar_historial()

except Exception as exc:
    st.error("No se pudieron cargar los archivos del modelo.")
    st.exception(exc)
    st.stop()


# ============================================================
# UTILIDADES
# ============================================================

def periodo_int_a_period(valor):
    return pd.Period(str(int(valor)), freq="M")


def period_a_int(periodo):
    return int(periodo.year * 100 + periodo.month)


def nombre_periodo(periodo):
    return f"{MESES_ES[periodo.month]} {periodo.year}"


def fecha_cierre_periodo(periodo):
    periodo_int = period_a_int(periodo)

    if periodo_int in CIERRES_2026:
        return pd.Timestamp(CIERRES_2026[periodo_int])

    return None


def dias_hasta_cierre(periodo):
    cierre = fecha_cierre_periodo(periodo)

    if cierre is None:
        # Respaldo: mediana histórica del mismo mes.
        aux = historial.copy()
        aux["MES_NUM"] = aux["PERIODO"].astype(int) % 100

        valores = pd.to_numeric(
            aux.loc[
                aux["MES_NUM"] == periodo.month,
                "DIAS_HASTA_CIERRE"
            ],
            errors="coerce"
        ).dropna()

        if not valores.empty:
            return float(valores.median())

        return float(
            pd.to_numeric(
                historial["DIAS_HASTA_CIERRE"],
                errors="coerce"
            ).dropna().median()
        )

    fin_mes = periodo.to_timestamp(how="end").normalize()

    return float((cierre - fin_mes).days)


def media_ultimos(df, columna, n=3):
    valores = pd.to_numeric(
        df[columna], errors="coerce"
    ).dropna().tail(n)

    if valores.empty:
        return np.nan

    return float(valores.mean())


def escenario_operativo(df_eess):
    """
    Valores operativos que todavía no se conocen en meses futuros.
    Se usa la media de los 3 últimos meses reales disponibles.
    """

    columnas = [
        "TOTAL_ATENCIONES",
        "N_REGISTRADORES",
        "N_UPS",
        "ATENCIONES_POR_REGISTRADOR",
        "PCT_ULTIMOS_3_DIAS",
        "PCT_DIA_CIERRE",
        "PCT_APP_1",
        "PCT_APP_3",
    ]

    salida = {}

    for col in columnas:
        salida[col] = media_ultimos(df_eess, col, 3)

    return salida


def construir_entrada(estado, fila_catalogo, periodo_objetivo):
    anterior = estado.iloc[-1]
    ultimos_3 = estado.tail(3)

    fila = {
        "DIAS_HASTA_CIERRE":
            dias_hasta_cierre(periodo_objetivo),

        "TOTAL_ATENCIONES_LAG1":
            float(anterior["TOTAL_ATENCIONES"]),

        "N_REGISTRADORES_LAG1":
            float(anterior["N_REGISTRADORES"]),

        "N_UPS_LAG1":
            float(anterior["N_UPS"]),

        "ATENCIONES_POR_REGISTRADOR_LAG1":
            float(anterior["ATENCIONES_POR_REGISTRADOR"]),

        "PCT_ATENCIONES_TARDIAS_LAG1":
            float(anterior["PCT_ATENCIONES_TARDIAS"]),

        "PCT_ATENCIONES_TARDIAS_MA3":
            float(
                pd.to_numeric(
                    ultimos_3["PCT_ATENCIONES_TARDIAS"],
                    errors="coerce"
                ).mean()
            ),

        "PCT_ULTIMOS_3_DIAS_LAG1":
            float(anterior["PCT_ULTIMOS_3_DIAS"]),

        "PCT_ULTIMOS_3_DIAS_MA3":
            float(
                pd.to_numeric(
                    ultimos_3["PCT_ULTIMOS_3_DIAS"],
                    errors="coerce"
                ).mean()
            ),

        "PCT_DIA_CIERRE_LAG1":
            float(anterior["PCT_DIA_CIERRE"]),

        "PCT_APP_1_LAG1":
            float(anterior["PCT_APP_1"]),

        "TOTAL_ATENCIONES_MA3":
            float(
                pd.to_numeric(
                    ultimos_3["TOTAL_ATENCIONES"],
                    errors="coerce"
                ).mean()
            ),

        "DESC_ESTAB":
            str(fila_catalogo["DESC_ESTAB"]),

        "RIS":
            str(fila_catalogo["RIS"]),

        "CAT_ESTAB":
            str(fila_catalogo["CAT_ESTAB"]),

        "DESC_DIST":
            str(fila_catalogo["DESC_DIST"]),

        "MES_CAT":
            str(periodo_objetivo.month).zfill(2),
    }

    entrada = pd.DataFrame([fila])

    # Respeta exactamente el orden de variables con el que fue entrenado.
    entrada = entrada[config["features"]]

    return entrada


def proyectar_hasta_diciembre(df_eess, fila_catalogo):
    """
    El primer mes posterior al último dato real es una predicción a 1 paso.

    Los meses posteriores son proyecciones recursivas:
    la predicción del mes previo alimenta PCT_ATENCIONES_TARDIAS_LAG1
    y los promedios móviles de los meses siguientes.
    """

    df_eess = (
        df_eess.sort_values("PERIODO")
        .copy()
        .reset_index(drop=True)
    )

    if df_eess.empty:
        raise ValueError(
            "El establecimiento no cuenta con historial suficiente."
        )

    ultimo_real = periodo_int_a_period(
        int(df_eess["PERIODO"].max())
    )

    inicio_proyeccion = ultimo_real + 1

    if inicio_proyeccion > PROYECCION_HASTA:
        return pd.DataFrame(), ultimo_real

    perfil = escenario_operativo(df_eess)

    estado = df_eess[
        [
            "PERIODO",
            "TOTAL_ATENCIONES",
            "N_REGISTRADORES",
            "N_UPS",
            "ATENCIONES_POR_REGISTRADOR",
            "PCT_ATENCIONES_TARDIAS",
            "PCT_ULTIMOS_3_DIAS",
            "PCT_DIA_CIERRE",
            "PCT_APP_1",
            "PCT_APP_3",
        ]
    ].copy()

    resultados = []

    periodo = inicio_proyeccion

    while periodo <= PROYECCION_HASTA:

        entrada = construir_entrada(
            estado,
            fila_catalogo,
            periodo
        )

        if entrada.isna().any().any():
            faltantes = entrada.columns[
                entrada.isna().any()
            ].tolist()

            raise ValueError(
                "Faltan datos para las variables: "
                + ", ".join(faltantes)
            )

        pred = float(modelo.predict(entrada)[0])
        pred = float(np.clip(pred, 0, 100))

        horizonte = (
            periodo.year - inicio_proyeccion.year
        ) * 12 + (
            periodo.month - inicio_proyeccion.month
        ) + 1

        resultados.append({
            "PERIODO":
                period_a_int(periodo),

            "FECHA":
                periodo.to_timestamp(),

            "PERIODO_TEXTO":
                nombre_periodo(periodo),

            "PREDICCION":
                pred,

            "HORIZONTE":
                horizonte,

            "TIPO":
                "Predicción a 1 mes"
                if horizonte == 1
                else "Proyección recursiva",
        })

        # Para el siguiente paso, la variable objetivo se actualiza
        # con la predicción. Las variables operativas desconocidas se
        # mantienen en el escenario base de los últimos 3 meses reales.
        nueva_fila = {
            "PERIODO":
                period_a_int(periodo),

            "TOTAL_ATENCIONES":
                perfil["TOTAL_ATENCIONES"],

            "N_REGISTRADORES":
                perfil["N_REGISTRADORES"],

            "N_UPS":
                perfil["N_UPS"],

            "ATENCIONES_POR_REGISTRADOR":
                perfil["ATENCIONES_POR_REGISTRADOR"],

            "PCT_ATENCIONES_TARDIAS":
                pred,

            "PCT_ULTIMOS_3_DIAS":
                perfil["PCT_ULTIMOS_3_DIAS"],

            "PCT_DIA_CIERRE":
                perfil["PCT_DIA_CIERRE"],

            "PCT_APP_1":
                perfil["PCT_APP_1"],

            "PCT_APP_3":
                perfil["PCT_APP_3"],
        }

        estado = pd.concat(
            [
                estado,
                pd.DataFrame([nueva_fila])
            ],
            ignore_index=True
        )

        periodo += 1

    return pd.DataFrame(resultados), ultimo_real


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    '<div class="titulo-principal">'
    'Predicción de retraso de digitación HISMINSA'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitulo-principal">
        Proyección del porcentaje de atenciones que podrían registrarse
        después de la fecha oficial de cierre.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Seleccionar establecimiento")

    ris_disponibles = sorted(
        catalogo["RIS"]
        .dropna()
        .unique()
        .tolist()
    )

    ris_sel = st.selectbox(
        "RIS",
        ris_disponibles
    )

    cat_ris = (
        catalogo[
            catalogo["RIS"] == ris_sel
        ]
        .sort_values("DESC_ESTAB")
        .copy()
    )

    estab_sel = st.selectbox(
        "Establecimiento",
        cat_ris["DESC_ESTAB"].tolist()
    )

    fila_catalogo = (
        cat_ris[
            cat_ris["DESC_ESTAB"] == estab_sel
        ]
        .iloc[0]
    )

    id_eess = int(
        fila_catalogo["ID_ESTABLECIMIENTO"]
    )

    st.markdown(
        """
        <div class="credito-sidebar">
            <b>Elaborado por:</b><br>
            Ing. Claudia Alocén
            <br><br>
            <b>Curso:</b><br>
            ENEI 2026-G4-612491-MACHINE LEARNING EN PRODUCCIÓN - DESPLIEGUE WEB
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HISTORIAL DEL ESTABLECIMIENTO
# ============================================================

df_eess = historial[
    historial["ID_ESTABLECIMIENTO"] == id_eess
].copy()

if df_eess.empty:
    st.warning(
        "El establecimiento seleccionado no cuenta con historial."
    )
    st.stop()


# ============================================================
# DATOS GENERALES
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "RIS",
    fila_catalogo["RIS"]
)

c2.metric(
    "Categoría",
    fila_catalogo["CAT_ESTAB"]
)

c3.metric(
    "Distrito",
    fila_catalogo["DESC_DIST"]
)

st.subheader(
    fila_catalogo["DESC_ESTAB"]
)


# ============================================================
# PROYECCIONES
# ============================================================

try:

    proyecciones, ultimo_real = proyectar_hasta_diciembre(
        df_eess,
        fila_catalogo
    )

except Exception as exc:

    st.error(
        "No fue posible generar la predicción."
    )

    st.exception(exc)
    st.stop()


if proyecciones.empty:

    st.info(
        "El historial ya llega hasta diciembre de 2026."
    )

else:

    inicio = periodo_int_a_period(
        int(proyecciones["PERIODO"].min())
    )

    fin = periodo_int_a_period(
        int(proyecciones["PERIODO"].max())
    )

    st.markdown(
        f"### Proyección {nombre_periodo(inicio)} "
        f"a {nombre_periodo(fin)}"
    )

    columnas = st.columns(
        len(proyecciones)
    )

    for col, (_, fila) in zip(
        columnas,
        proyecciones.iterrows()
    ):

        col.metric(
            fila["PERIODO_TEXTO"],
            f"{fila['PREDICCION']:.2f}%"
        )

        if int(fila["HORIZONTE"]) == 1:
            col.caption("Predicción a 1 mes")
        else:
            col.caption(
                f"Proyección a {int(fila['HORIZONTE'])} meses"
            )

    st.markdown(
        """
        <div class="bloque-info">
            <b>Interpretación:</b>
            cada porcentaje representa la proporción estimada de
            atenciones del periodo que podrían registrarse después
            de su fecha oficial de cierre.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(proyecciones) > 1:
        st.markdown(
            """
            <div class="bloque-advertencia">
                <b>Importante:</b>
                el primer mes es una predicción directa basada en el
                último mes real disponible. Los meses posteriores son
                proyecciones recursivas y utilizan las predicciones
                anteriores junto con el comportamiento operativo reciente.
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # TABLA RESUMEN
    # ========================================================

    tabla = proyecciones[
        [
            "PERIODO_TEXTO",
            "PREDICCION",
            "TIPO"
        ]
    ].copy()

    tabla.columns = [
        "Periodo",
        "% estimado de atenciones tardías",
        "Tipo de estimación"
    ]

    tabla[
        "% estimado de atenciones tardías"
    ] = tabla[
        "% estimado de atenciones tardías"
    ].round(2)

    st.dataframe(
        tabla,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # GRÁFICO HISTÓRICO + PROYECCIÓN
    # ========================================================

    st.markdown(
        "### Evolución histórica y proyección"
    )

    historico_plot = df_eess[
        [
            "PERIODO",
            "PCT_ATENCIONES_TARDIAS"
        ]
    ].copy()

    historico_plot["FECHA"] = pd.PeriodIndex(
        historico_plot["PERIODO"]
        .astype(int)
        .astype(str),
        freq="M"
    ).to_timestamp()

    historico_plot["VALOR"] = pd.to_numeric(
        historico_plot["PCT_ATENCIONES_TARDIAS"],
        errors="coerce"
    )

    historico_plot["SERIE"] = "Histórico real"

    historico_plot["ETIQUETA"] = (
        historico_plot["FECHA"]
        .dt.strftime("%Y-%m")
    )

    historico_plot = historico_plot[
        [
            "FECHA",
            "ETIQUETA",
            "VALOR",
            "SERIE"
        ]
    ].dropna()


    proy_plot = proyecciones[
        [
            "FECHA",
            "PERIODO_TEXTO",
            "PREDICCION"
        ]
    ].copy()

    proy_plot = proy_plot.rename(
        columns={
            "PERIODO_TEXTO": "ETIQUETA",
            "PREDICCION": "VALOR"
        }
    )

    proy_plot["SERIE"] = "Proyección"

    proy_plot = proy_plot[
        [
            "FECHA",
            "ETIQUETA",
            "VALOR",
            "SERIE"
        ]
    ]


    # Punto ancla: último valor real para conectar visualmente
    # la serie histórica con la proyección.
    ultimo_real_plot = (
        historico_plot
        .sort_values("FECHA")
        .tail(1)
        .copy()
    )

    ancla = ultimo_real_plot.copy()
    ancla["SERIE"] = "Proyección"

    grafico_df = pd.concat(
        [
            historico_plot,
            ancla,
            proy_plot
        ],
        ignore_index=True
    )


    lineas = (
        alt.Chart(grafico_df)
        .mark_line(
            point=True,
            strokeWidth=2.5
        )
        .encode(
            x=alt.X(
                "FECHA:T",
                title="Periodo",
                axis=alt.Axis(
                    format="%Y-%m",
                    labelAngle=-45
                )
            ),

            y=alt.Y(
                "VALOR:Q",
                title="% de atenciones tardías",
                scale=alt.Scale(zero=True)
            ),

            color=alt.Color(
                "SERIE:N",
                title="Serie",
                scale=alt.Scale(
                    domain=[
                        "Histórico real",
                        "Proyección"
                    ],
                    range=[
                        "#0096ba",
                        "#ef4444"
                    ]
                )
            ),

            strokeDash=alt.StrokeDash(
                "SERIE:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "Histórico real",
                        "Proyección"
                    ],
                    range=[
                        [1, 0],
                        [7, 4]
                    ]
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "ETIQUETA:N",
                    title="Periodo"
                ),

                alt.Tooltip(
                    "SERIE:N",
                    title="Serie"
                ),

                alt.Tooltip(
                    "VALOR:Q",
                    title="% tardías",
                    format=".2f"
                ),
            ]
        )
        .properties(
            height=370
        )
    )

    st.altair_chart(
        lineas,
        width="stretch"
    )


# ============================================================
# DESEMPEÑO DEL MODELO
# ============================================================

st.markdown("### Desempeño del modelo")

m1, m2, m3 = st.columns(3)

r2 = float(config.get("r2_test", np.nan))
mae = float(config.get("mae_test", np.nan))
rmse = float(config.get("rmse_test", np.nan))

with m1:
    if not np.isnan(r2):
        st.metric(
            "R²",
            f"{r2:.3f}"
        )
        st.caption(
            f"El modelo explica aproximadamente {r2 * 100:.1f}% "
            f"de la variabilidad observada."
        )

with m2:
    if not np.isnan(mae):
        st.metric(
            "MAE",
            f"{mae:.2f} pp"
        )
        st.caption(
            "Error absoluto medio de las predicciones en el test temporal."
        )

with m3:
    if not np.isnan(rmse):
        st.metric(
            "RMSE",
            f"{rmse:.2f} pp"
        )
        st.caption(
            "Error que penaliza con mayor peso las diferencias grandes."
        )


# ============================================================
# INFORMACIÓN TÉCNICA
# ============================================================

with st.expander(
    "Información técnica del modelo"
):

    ultimo_periodo = int(
        df_eess["PERIODO"].max()
    )

    st.write(
        f"Último periodo real disponible para el establecimiento: "
        f"**{ultimo_periodo}**."
    )

    if "rmse_test" in config:
        st.write(
            f"RMSE de test: **{config['rmse_test']:.2f} "
            f"puntos porcentuales**."
        )

    if "mae_test" in config:
        st.write(
            f"MAE de test: **{config['mae_test']:.2f} "
            f"puntos porcentuales**."
        )

    if "r2_test" in config:
        st.write(
            f"R² de test: **{config['r2_test']:.3f}**."
        )

    st.write(
        "El primer periodo futuro se estima con antecedentes reales. "
        "Cuando se proyectan meses adicionales, el porcentaje tardío "
        "estimado del mes previo pasa a formar parte de los antecedentes "
        "del siguiente mes."
    )


st.caption(
    "Modelo académico de regresión para el análisis de oportunidad "
    "de digitación HISMINSA."
)
