# ------------------------------
# Importação das bibliotecas necessárias
# ------------------------------
import os                      # Para manipulação de caminhos de arquivos
from io import BytesIO         # Para manipular dados em memória (buffer de bytes)
import numpy as np             # Biblioteca para cálculos numéricos
import pandas as pd            # Biblioteca para manipulação de DataFrames (tabelas)
import joblib                  # Para carregar modelos e scalers salvos em arquivos .pkl
import streamlit as st         # Para criação da interface web interativa
import matplotlib.pyplot as plt # Para geração de gráficos

# ------------------------------
# Configuração inicial da página Streamlit
# ------------------------------
st.set_page_config(page_title="Análise de Contaminação do Solo", layout="wide")
st.title("🧪 Análise de Contaminação do Solo")
st.caption("Envie CSV/XLSX com os dados das amostras.")

# ------------------------------
# Caminhos padrão para o modelo e scaler
# ------------------------------
DEFAULT_MODEL_PATHS = [
    os.path.join("/mnt/data", "modelo_knn.pkl"), # Caminho absoluto (Linux/Mount)
    "modelo_knn.pkl",                           # Caminho relativo (diretório atual)
]
DEFAULT_SCALER_PATHS = [
    os.path.join("/mnt/data", "scaler.pkl"),
    "scaler.pkl",
]

# ---------------------------------------
# Funções utilitárias
# ---------------------------------------

# Função para carregar um arquivo .pkl (modelo/scaler) do primeiro caminho existente
def load_joblib_first_found(paths):
    last_err = None
    for p in paths:
        if os.path.exists(p):           # Verifica se o arquivo existe
            return joblib.load(p)       # Carrega e retorna o objeto salvo

# Função para ler arquivos CSV ou XLSX enviados pelo usuário
def read_any_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):           # Se for CSV
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):        # Se for Excel
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato não suportado. Envie .csv ou .xlsx.")

# Função para plotar histogramas das variáveis numéricas
def plot_hist_grid(df: pd.DataFrame, cols, bins: int, title: str):
    if len(cols) == 0:  # Caso não existam colunas numéricas
        st.info("Não há colunas numéricas para plotar.")
        return
    ncols = 3 if len(cols) >= 3 else len(cols)   # Define no máximo 3 gráficos por linha
    nrows = int(np.ceil(len(cols) / ncols))      # Calcula número de linhas
    fig = plt.figure(figsize=(5*ncols, 3*nrows)) # Tamanho da figura
    for i, col in enumerate(cols, start=1):      # Itera pelas colunas numéricas
        ax = fig.add_subplot(nrows, ncols, i)    # Cria subplot
        ax.hist(df[col].dropna(), bins=bins, edgecolor="black", alpha=0.8)
        ax.set_title(col)                        # Nome da variável no título
        ax.grid(True, linestyle="--", alpha=0.3) # Grelha leve
    fig.suptitle(title)
    fig.tight_layout()
    st.pyplot(fig)

# Função para plotar histogramas comparando classes (0 = não contaminado, 1 = contaminado)
def plot_hist_by_class(df: pd.DataFrame, cols, label_col: str, bins: int, title: str):
    if len(cols) == 0:
        st.info("Não há colunas numéricas para plotar.")
        return
    ncols = 3 if len(cols) >= 3 else len(cols)
    nrows = int(np.ceil(len(cols) / ncols))
    fig = plt.figure(figsize=(5*ncols, 3*nrows))
    for i, col in enumerate(cols, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        # Separa valores de acordo com a classe prevista
        g0 = df[df[label_col] == 0][col].dropna().values
        g1 = df[df[label_col] == 1][col].dropna().values
        if g0.size > 0:
            ax.hist(g0, bins=bins, edgecolor="black", alpha=0.6, label="não contaminado (0)")
        if g1.size > 0:
            ax.hist(g1, bins=bins, edgecolor="black", alpha=0.6, label="contaminado (1)")
        ax.set_title(col)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    st.pyplot(fig)

# Função para gerar Excel com duas abas: predições e resumo
def write_excel_bytes(df_main: pd.DataFrame, df_resumo: pd.DataFrame) -> bytes:
    output = BytesIO()
    try:
        # Primeiro tenta com xlsxwriter
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_main.to_excel(writer, index=False, sheet_name="Predicoes")
            df_resumo.to_excel(writer, index=False, sheet_name="Resumo")
    except ModuleNotFoundError:
        # Fallback para openpyxl se xlsxwriter não estiver instalado
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_main.to_excel(writer, index=False, sheet_name="Predicoes")
            df_resumo.to_excel(writer, index=False, sheet_name="Resumo")
    return output.getvalue()

# ---------------------------------------
# Carregamento do modelo e scaler
# ---------------------------------------
try:
    model = load_joblib_first_found(DEFAULT_MODEL_PATHS)
except Exception as e:
    st.error(f"Não foi possível carregar o modelo: {e}")
    st.stop()

try:
    scaler = load_joblib_first_found(DEFAULT_SCALER_PATHS)
except Exception as e:
    st.error(f"Não foi possível carregar o scaler: {e}")
    st.stop()

# Verifica se o scaler preserva os nomes das features usadas no treino
if not hasattr(scaler, "feature_names_in_"):
    st.error("O scaler não expõe `feature_names_in_`. Refit o scaler no treino (sklearn>=1.0) para preservar nomes de colunas.")
    st.stop()

expected_cols = list(scaler.feature_names_in_) # Lista de colunas esperadas

st.divider()

# ---------------------------------------
# Upload dos dados (CSV/XLSX)
# ---------------------------------------
st.subheader("Upload dos Dados de Entrada (CSV ou XLSX)")
data_file = st.file_uploader(
    "Faça o upload do arquivo",
    type=["csv", "xlsx"],
    key="data_file"
)

if data_file is None:
    st.stop()

# Leitura do arquivo enviado
try:
    data_raw = read_any_table(data_file)
except Exception as e:
    st.error(f"Erro ao ler arquivo: {e}")
    st.stop()

if data_raw.empty:
    st.error("Arquivo está vazio.")
    st.stop()

# Verifica se todas as features necessárias estão presentes
missing = [c for c in expected_cols if c not in data_raw.columns]
if missing:
    st.error(f"Features ausentes no arquivo (necessárias pelo scaler/treino): {missing}")
    st.stop()

# Mantém somente as colunas esperadas (ignora extras)
X = data_raw[expected_cols].copy()

# Garante que todas as colunas sejam numéricas
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")

# Imputação simples com mediana se houver valores ausentes
if X.isna().any().any():
    st.warning("Foram encontrados valores ausentes. Aplicando imputação simples (mediana) antes da padronização.")
    X = X.fillna(X.median(numeric_only=True))

# Exibe pré-visualização dos dados
st.markdown("### Pré-visualização dos dados")
st.dataframe(data_raw.head(100), use_container_width=True)

# ---------------------------------------
# Histogramas – dados do usuário
# ---------------------------------------
st.markdown("### Histogramas – Dados enviados")
bins = st.slider("Número de bins", min_value=10, max_value=60, value=30, step=5)
num_cols = X.columns.tolist()
plot_hist_grid(X, num_cols, bins, "Histogramas das variáveis enviadas")

st.divider()

# ---------------------------------------
# Classificação (scaler aplicado automaticamente)
# ---------------------------------------
run = st.button("🚀 Classificar", type="primary")
if not run:
    st.stop()

# Aplica o scaler (padronização) nas colunas do treino
try:
    X_scaled = pd.DataFrame(
        scaler.transform(X[expected_cols]),
        columns=expected_cols,
        index=X.index
    )
except Exception as e:
    st.error(f"Falha ao aplicar scaler: {e}")
    st.stop()

# Realiza a predição com o modelo treinado
try:
    y_pred = model.predict(X_scaled)
except Exception as e:
    st.error(f"Erro ao executar predição: {e}")
    st.stop()

# Adiciona colunas de resultados no DataFrame original
pred_df = data_raw.copy()
pred_df["pred"] = y_pred
pred_df["pred_label"] = np.where(pred_df["pred"] == 1, "contaminado", "não contaminado")

# Mostra resultados na tela
st.markdown("### ✅ Resultado")
st.dataframe(pred_df.head(100), use_container_width=True)

# Métricas simples de contagem
c1, c2 = st.columns(2)
with c1:
    st.metric("Não contaminado (0)", int((pred_df["pred"] == 0).sum()))
with c2:
    st.metric("Contaminado (1)", int((pred_df["pred"] == 1).sum()))

st.divider()

# ---------------------------------------
# Histogramas comparando classes previstas
# ---------------------------------------
st.markdown("### Histogramas comparativos: Contaminado × Não contaminado")
plot_hist_by_class(pred_df, num_cols, label_col="pred", bins=bins, title="Distribuições por classe prevista")

# ---------------------------------------
# Download em XLSX com predições e resumo
# ---------------------------------------
resumo = pd.DataFrame({
    "classe": ["não contaminado (0)", "contaminado (1)"],
    "quantidade": [(pred_df["pred"] == 0).sum(), (pred_df["pred"] == 1).sum()]
})

# Gera arquivo Excel em memória
xlsx_bytes = write_excel_bytes(pred_df, resumo)

# Botão de download do arquivo Excel
st.download_button(
    "💾 Download do resultado",
    data=xlsx_bytes,
    file_name="predicoes_contaminacao.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
