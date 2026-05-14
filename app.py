from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "clientes_churn_ficticio.csv"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "churn_preprocessor.pkl"
COMPARATIVO_PATH = ROOT_DIR / "outputs" / "comparativo_modelos.csv"
ARTIFACTS_PATH = ROOT_DIR / "outputs" / "model_artifacts.joblib"

COLOR_CHURN = "#EF4444"
COLOR_ACTIVE = "#22C55E"
COLOR_MEDIUM = "#FACC15"
COLOR_PRIMARY = "#00B4D8"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#E2E8F0"
COLOR_BG = "#FFFFFF"

FEATURE_COLUMNS = [
    "plano",
    "tempo_como_cliente_meses",
    "numero_produtos",
    "tickets_abertos_ultimo_mes",
    "tickets_resolvidos",
    "nps_score",
    "uso_medio_mensal_horas",
    "atraso_pagamento_dias",
    "desconto_recebido",
    "canal_aquisicao",
    "regiao",
]

PLANOS = ["Starter", "Pro", "Enterprise"]
CANAIS = ["Indicação", "Google Ads", "LinkedIn", "Evento", "Inbound"]
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
PAGES = ["Visão Geral", "Análise de Risco", "Preditor Individual", "Análise em Lote", "Performance do Modelo"]


st.set_page_config(page_title="Nexus Churn Intelligence", page_icon="N", layout="wide")
sns.set_theme(style="whitegrid")


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {COLOR_PRIMARY};
            --churn: {COLOR_CHURN};
            --active: {COLOR_ACTIVE};
            --medium: {COLOR_MEDIUM};
            --text: {COLOR_TEXT};
            --muted: {COLOR_MUTED};
            --border: {COLOR_BORDER};
            --bg: {COLOR_BG};
        }}
        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {{
            display: none;
        }}
        [data-testid="stSidebar"] {{
            background: #F8FAFC;
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: var(--text);
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            border-radius: 8px;
            padding: .5rem .65rem;
            margin-bottom: .15rem;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
            background: #E0F7FC;
            border-left: 4px solid var(--primary);
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] input {{
            accent-color: var(--primary);
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }}
        h1, h2, h3 {{
            letter-spacing: 0;
            color: var(--text);
        }}
        h1 {{
            font-size: 2rem;
            line-height: 1.15;
            margin-bottom: .25rem;
        }}
        h2 {{
            font-size: 1.25rem;
            margin-top: .4rem;
        }}
        .nexus-subtitle {{
            color: var(--muted);
            font-size: .98rem;
            margin: 0 0 1rem;
        }}
        .metric-card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px 18px 16px;
            background: #FFFFFF;
            min-height: 112px;
        }}
        .metric-label {{
            color: var(--muted);
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
            margin-bottom: .55rem;
        }}
        .metric-value {{
            color: var(--text);
            font-size: 1.9rem;
            line-height: 1;
            font-weight: 760;
        }}
        .metric-note {{
            color: var(--muted);
            font-size: .86rem;
            margin-top: .5rem;
        }}
        .risk-panel {{
            border: 1px solid var(--border);
            border-left-width: 8px;
            border-radius: 8px;
            padding: 1.2rem 1.25rem;
            background: #FFFFFF;
        }}
        .risk-probability {{
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1;
            color: var(--text);
        }}
        .risk-class {{
            font-size: 1.05rem;
            font-weight: 760;
            margin-top: .4rem;
        }}
        .risk-action {{
            color: var(--muted);
            margin-top: .5rem;
        }}
        .risk-meter {{
            height: 8px;
            width: 100%;
            background: #E2E8F0;
            border-radius: 999px;
            overflow: hidden;
            margin-top: .7rem;
        }}
        .risk-meter-fill {{
            height: 100%;
            border-radius: 999px;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--text);
        }}
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCheckbox"] label p,
        [data-testid="stSlider"] label p {{
            color: var(--text) !important;
            font-weight: 650;
        }}
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            background: #FFFFFF !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] span {{
            color: var(--text) !important;
        }}
        [data-testid="stDateInput"] input {{
            color: var(--text) !important;
        }}
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stFormSubmitButton"] button {{
            border-radius: 7px;
            border: 1px solid var(--primary);
            background: var(--primary);
            color: white;
            font-weight: 700;
        }}
        div[data-testid="stFileUploader"] section {{
            border-radius: 8px;
            border-color: var(--border);
        }}
        .dataframe {{
            border: 1px solid var(--border);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["data_cadastro"])
    return df


@st.cache_resource
def load_model_bundle() -> tuple[Any, Any]:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@st.cache_data
def load_model_artifacts() -> dict[str, Any]:
    if ARTIFACTS_PATH.exists():
        return joblib.load(ARTIFACTS_PATH)
    return {}


@st.cache_data
def load_comparativo() -> pd.DataFrame:
    if COMPARATIVO_PATH.exists():
        return pd.read_csv(COMPARATIVO_PATH)
    return pd.DataFrame()


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<p class='nexus-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def render_metric(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_churn_labels(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["status"] = work["churn"].map({0: "Ativo", 1: "Churn"})
    return work


def fig_churn_distribution(df: pd.DataFrame):
    work = prepare_churn_labels(df)
    counts = work["status"].value_counts().reindex(["Ativo", "Churn"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(counts.index, counts.values, color=[COLOR_ACTIVE, COLOR_CHURN], width=0.54)
    ax.set_title("Distribuição de churn", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Clientes")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar in bars:
        value = int(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:,}".replace(",", "."), ha="center", fontweight="bold")
    return fig


def fig_churn_rate_by_category(df: pd.DataFrame, category: str, title: str, color: str = COLOR_CHURN):
    rates = df.groupby(category)["churn"].mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    sns.barplot(x=rates.values, y=rates.index, ax=ax, color=color)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Taxa de churn (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(35, rates.max() * 1.25))
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for i, value in enumerate(rates.values):
        ax.text(value + 0.6, i, f"{value:.1f}%", va="center", fontsize=10, fontweight="bold")
    return fig


def fig_boxplot(df: pd.DataFrame, variable: str, title: str, show_points: bool = False):
    work = prepare_churn_labels(df)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    palette = {"Ativo": COLOR_ACTIVE, "Churn": COLOR_CHURN}
    sns.boxplot(data=work, x="status", y=variable, ax=ax, palette=palette, width=0.5, fliersize=2)
    if show_points:
        sns.stripplot(data=work.sample(min(len(work), 600), random_state=42), x="status", y=variable, ax=ax, color="#0F172A", alpha=0.18, size=2)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(variable.replace("_", " "))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def fig_confusion_matrix(confusion_matrix_values):
    labels = ["Ativo", "Churn"]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sns.heatmap(confusion_matrix_values, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title("Matriz de confusão", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    return fig


def fig_roc_curve(roc_curve_data: dict[str, list[float]]):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(roc_curve_data["fpr"], roc_curve_data["tpr"], color=COLOR_PRIMARY, linewidth=2.5, label="Modelo salvo")
    ax.plot([0, 1], [0, 1], color=COLOR_MUTED, linestyle="--", linewidth=1.2, label="Aleatório")
    ax.set_title("Curva ROC", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Falso positivo")
    ax.set_ylabel("Verdadeiro positivo")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_feature_importance(artifacts: dict[str, Any]):
    importances = artifacts.get("feature_importance")
    names = artifacts.get("feature_names")
    if not importances or not names:
        return None

    importance_df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5.2))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color=COLOR_PRIMARY)
    ax.set_title("Feature importance", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importância")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def normalize_discount(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)

    true_values = {"sim", "s", "yes", "y", "true", "1", "recebeu", "com desconto"}
    return series.fillna("não").astype(str).str.strip().str.lower().isin(true_values).astype(int)


def sanitize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "desconto_recebido" in prepared.columns:
        prepared["desconto_recebido"] = normalize_discount(prepared["desconto_recebido"])

    numeric_columns = [
        "tempo_como_cliente_meses",
        "numero_produtos",
        "tickets_abertos_ultimo_mes",
        "tickets_resolvidos",
        "nps_score",
        "uso_medio_mensal_horas",
        "atraso_pagamento_dias",
        "desconto_recebido",
    ]
    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared


def predict_probabilities(input_df: pd.DataFrame) -> np.ndarray:
    model, preprocessor = load_model_bundle()
    prepared = sanitize_feature_frame(input_df)
    transformed = preprocessor.transform(prepared[FEATURE_COLUMNS])
    return model.predict_proba(transformed)[:, 1]


def classify_risk(probability: float) -> tuple[str, str, str]:
    if probability > 0.70:
        return "Alto", COLOR_CHURN, "Acionar gerente de conta imediatamente."
    if probability >= 0.40:
        return "Médio", COLOR_MEDIUM, "Enviar campanha de retenção."
    return "Baixo", COLOR_ACTIVE, "Monitorar no próximo ciclo."


def page_overview(df: pd.DataFrame) -> None:
    render_header("Nexus Churn Intelligence", "Monitoramento de risco de cancelamento para a operação comercial da Nexus.")

    churn_rate = df["churn"].mean() * 100
    canal_rates = df.groupby("canal_aquisicao")["churn"].mean().sort_values(ascending=False)
    worst_channel = canal_rates.index[0]
    worst_channel_rate = canal_rates.iloc[0] * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("Total de clientes", f"{len(df):,}".replace(",", "."), "Base fictícia gerada com seed 42")
    with col2:
        render_metric("Taxa de churn", f"{churn_rate:.1f}%", "Proporção de cancelamentos")
    with col3:
        render_metric("Clientes em churn", f"{int(df['churn'].sum()):,}".replace(",", "."), "Classe positiva do modelo")
    with col4:
        render_metric("Canal com maior perda", worst_channel, f"{worst_channel_rate:.1f}% de churn")

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.pyplot(fig_churn_distribution(df), clear_figure=True, use_container_width=True)
    with right:
        st.pyplot(fig_churn_rate_by_category(df, "plano", "Churn por plano", COLOR_PRIMARY), clear_figure=True, use_container_width=True)

    st.pyplot(fig_churn_rate_by_category(df, "canal_aquisicao", "Churn por canal de aquisição", COLOR_CHURN), clear_figure=True, use_container_width=True)


def page_risk_analysis(df: pd.DataFrame) -> None:
    render_header("Análise de Risco", "Segmentos com maior risco e sinais comportamentais associados ao cancelamento.")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.pyplot(fig_churn_rate_by_category(df, "plano", "Churn por plano", COLOR_PRIMARY), clear_figure=True, use_container_width=True)
    with col2:
        st.pyplot(fig_churn_rate_by_category(df, "canal_aquisicao", "Churn por canal", COLOR_CHURN), clear_figure=True, use_container_width=True)

    st.pyplot(fig_churn_rate_by_category(df, "regiao", "Churn por região", COLOR_PRIMARY), clear_figure=True, use_container_width=True)

    st.divider()
    variable_options = {
        "NPS": ("nps_score", "NPS segmentado por churn"),
        "Uso médio mensal": ("uso_medio_mensal_horas", "Uso médio mensal segmentado por churn"),
    }
    control_col, toggle_col = st.columns([0.7, 0.3])
    with control_col:
        selected_label = st.radio("Variável do boxplot", list(variable_options.keys()), horizontal=True)
    with toggle_col:
        show_points = st.checkbox("Exibir amostra de pontos", value=False)
    selected_variable, title = variable_options[selected_label]
    st.pyplot(fig_boxplot(df, selected_variable, title, show_points), clear_figure=True, use_container_width=True)


def page_individual_predictor() -> None:
    render_header("Preditor Individual", "Estimativa de probabilidade de churn para um cliente específico.")

    with st.form("individual_prediction_form"):
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            cliente_id = st.text_input("Cliente ID", value="NOVO-001")
            plano = st.selectbox("Plano", PLANOS, index=1)
        with row1_col2:
            data_cadastro = st.date_input("Data de cadastro")
            canal = st.selectbox("Canal de aquisição", CANAIS, index=1)
        with row1_col3:
            regiao = st.selectbox("Região", REGIOES, index=3)
            desconto = st.checkbox("Desconto recebido", value=False)

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            tempo = st.slider("Tempo como cliente (meses)", 1, 72, 12)
            numero_produtos = st.slider("Número de produtos", 1, 5, 2)
        with col2:
            tickets_abertos = st.slider("Tickets abertos no último mês", 0, 15, 5)
            tickets_resolvidos = st.slider("Tickets resolvidos", 0, 15, 2)
        with col3:
            nps = st.slider("NPS", 0, 100, 35)
            uso_medio = st.slider("Uso médio mensal (horas)", 0.5, 80.0, 10.0, step=0.5)
            atraso = st.slider("Atraso de pagamento (dias)", 0, 45, 10)

        submitted = st.form_submit_button("Calcular Risco")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "cliente_id": cliente_id,
                    "data_cadastro": data_cadastro,
                    "plano": plano,
                    "tempo_como_cliente_meses": tempo,
                    "numero_produtos": numero_produtos,
                    "tickets_abertos_ultimo_mes": tickets_abertos,
                    "tickets_resolvidos": min(tickets_resolvidos, tickets_abertos),
                    "nps_score": nps,
                    "uso_medio_mensal_horas": uso_medio,
                    "atraso_pagamento_dias": atraso,
                    "desconto_recebido": int(desconto),
                    "canal_aquisicao": canal,
                    "regiao": regiao,
                }
            ]
        )
        probability = float(predict_probabilities(input_df)[0])
        risk_class, risk_color, recommendation = classify_risk(probability)

        st.markdown(
            f"""
            <div class="risk-panel" style="border-left-color: {risk_color};">
                <div class="metric-label">Risco de Churn</div>
                <div class="risk-probability">{probability * 100:.1f}%</div>
                <div class="risk-class" style="color: {risk_color};">Classificação: {risk_class}</div>
                <div class="risk-action">{recommendation}</div>
                <div class="risk-meter"><div class="risk-meter-fill" style="width: {probability * 100:.1f}%; background: {risk_color};"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_batch_analysis() -> None:
    render_header("Análise em Lote", "Classificação de múltiplos clientes a partir de um arquivo CSV.")

    uploaded_file = st.file_uploader("Upload de CSV com novos clientes", type=["csv"])
    st.caption("O arquivo deve conter as colunas de entrada do modelo. Colunas como cliente_id e data_cadastro são preservadas no resultado.")

    if uploaded_file is None:
        return

    batch_df = pd.read_csv(uploaded_file)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in batch_df.columns]
    if missing_columns:
        st.error("Colunas ausentes no CSV: " + ", ".join(missing_columns))
        return

    probabilities = predict_probabilities(batch_df)
    result = batch_df.copy()
    result["probabilidade_churn"] = probabilities
    result["risco_churn_pct"] = (probabilities * 100).round(2)
    result["classificacao_risco"] = [classify_risk(float(probability))[0] for probability in probabilities]
    result["recomendacao"] = [classify_risk(float(probability))[2] for probability in probabilities]

    st.dataframe(result, use_container_width=True, hide_index=True)

    csv_bytes = result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Baixar resultado",
        data=csv_bytes,
        file_name="clientes_churn_classificados.csv",
        mime="text/csv",
    )


def page_model_performance() -> None:
    render_header("Performance do Modelo", "Métricas do modelo salvo e diagnóstico de classificação.")

    artifacts = load_model_artifacts()
    comparison = load_comparativo()

    if comparison.empty or not artifacts:
        st.warning("Execute o notebook para gerar métricas, modelo e artefatos de avaliação.")
        return

    best_model_name = artifacts.get("best_model_name", "Modelo salvo")
    best_metrics = artifacts.get("metrics", {})

    st.subheader(f"Modelo salvo: {best_model_name}")
    metric_cols = st.columns(5)
    metric_labels = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    for column, (metric_key, label) in zip(metric_cols, metric_labels):
        with column:
            value = float(best_metrics.get(metric_key, 0))
            render_metric(label, f"{value:.3f}")

    st.dataframe(comparison.sort_values("f1", ascending=False), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.pyplot(fig_confusion_matrix(np.array(artifacts["confusion_matrix"])), clear_figure=True, use_container_width=True)
    with col2:
        st.pyplot(fig_roc_curve(artifacts["roc_curve"]), clear_figure=True, use_container_width=True)

    importance_fig = fig_feature_importance(artifacts)
    if importance_fig is not None:
        st.pyplot(importance_fig, clear_figure=True, use_container_width=True)
    else:
        st.info("Feature importance não disponível para o tipo do modelo salvo.")


def ensure_artifacts_available() -> None:
    required_paths = [DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH]
    missing = [path.name for path in required_paths if not path.exists()]
    if missing:
        st.error(
            "Artefatos ausentes: "
            + ", ".join(missing)
            + ". Execute `python gerar_base.py` e depois o notebook antes de abrir o app."
        )
        st.stop()


def main() -> None:
    apply_theme()
    ensure_artifacts_available()
    df = load_data()

    with st.sidebar:
        st.title("Nexus")
        st.caption("Churn Intelligence")
        page = st.radio("Navegação", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("Cores de risco")
        st.markdown(
            f"""
            <div style="display:flex; gap:.5rem; align-items:center; margin:.35rem 0;"><span style="width:12px;height:12px;background:{COLOR_ACTIVE};display:inline-block;border-radius:3px;"></span> Baixo</div>
            <div style="display:flex; gap:.5rem; align-items:center; margin:.35rem 0;"><span style="width:12px;height:12px;background:{COLOR_MEDIUM};display:inline-block;border-radius:3px;"></span> Médio</div>
            <div style="display:flex; gap:.5rem; align-items:center; margin:.35rem 0;"><span style="width:12px;height:12px;background:{COLOR_CHURN};display:inline-block;border-radius:3px;"></span> Alto</div>
            """,
            unsafe_allow_html=True,
        )

    if page == "Visão Geral":
        page_overview(df)
    elif page == "Análise de Risco":
        page_risk_analysis(df)
    elif page == "Preditor Individual":
        page_individual_predictor()
    elif page == "Análise em Lote":
        page_batch_analysis()
    elif page == "Performance do Modelo":
        page_model_performance()


if __name__ == "__main__":
    main()
