# Streamlit – Monitor de Carteiras Recomendadas (B3 + global)
# Autor: você ;)
# Deploy: suba este arquivo no GitHub e conecte em streamlit.app
# Requisitos: streamlit, yfinance, pandas, numpy, plotly, pytz, python-dateutil

import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from datetime import datetime, date, timedelta

# HTTP opcional para carregar config remota
try:
    import requests  # adicione 'requests' ao requirements.txt
except Exception:  # fallback quando não instalado
    requests = None
from pathlib import Path

# =============================
# Configuração básica da página
# =============================
st.set_page_config(
    page_title="Monitor de Carteiras",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Compactar um pouco o editor e as tabelas */
    .stDataFrame, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1n76uvr {font-size: 0.92rem;}
    .small {font-size: 0.85rem; color:#555}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================
# Utilidades gerais
# ================

BENCH_PRESETS = {
    "Ibovespa (^BVSP)": "^BVSP",
    "BOVA11 (ETF)": "BOVA11.SA",
    "S&P 500 (^GSPC)": "^GSPC",
    "NASDAQ 100 (^NDX)": "^NDX",
    "Dólar/Real (USDBRL)": "USDBRL=X",
}

DATE_TZ = "America/Sao_Paulo"

# =============================
# Persistência simples em disco
# =============================
STATE_PATH = Path.home() / ".streamlit_carteiras_config.json"

def _save_persisted_portfolios(portfolios_state: Dict[str, "Portfolio"]):
    try:
        data = {pid: p.to_dict() for pid, p in portfolios_state.items()}
        payload = {
            "portfolios": data,
            "_meta": {
                "logo_src": st.session_state.get("logo_src", "assets/icone_completo_2022_fundo_titanio.png"),
                "config_url": st.session_state.get("config_url", ""),
            },
        }
        STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _load_persisted_portfolios() -> Dict[str, "Portfolio"] | None:
    try:
        if not STATE_PATH.exists():
            return None
        raw_text = STATE_PATH.read_text(encoding="utf-8")
        if not raw_text.strip():
            return None
        raw = json.loads(raw_text)

        if isinstance(raw, dict) and "portfolios" in raw:
            raw_ports = raw.get("portfolios", {})
            meta = raw.get("_meta", {})
            if isinstance(meta, dict):
                if meta.get("logo_src"):
                    st.session_state.logo_src = meta.get("logo_src")
                if meta.get("config_url") is not None:
                    st.session_state.config_url = meta.get("config_url")
        else:
            raw_ports = raw

        loaded: Dict[str, Portfolio] = {}
        for k, v in raw_ports.items():
            loaded[str(k)] = Portfolio(
                name=v.get("name", f"Carteira {k}"),
                tickers=v.get("tickers", []),
                benchmark=v.get("benchmark", "^BVSP"),
            )
        return loaded
    except Exception:
        return None

# =============================
# Config remota (GitHub Raw / Gist / URL pública JSON)
# =============================

def _fetch_remote_config(url: str) -> Dict[str, "Portfolio"] | None:
    if not url:
        return None
    try:
        if requests is None:
            st.warning("Para carregar configuração remota, adicione 'requests' ao requirements.txt")
            return None
        r = requests.get(url, timeout=10)
        if not r.ok:
            return None
        raw = r.json()
        # Formato com 'portfolios' ou dict direto
        raw_ports = raw.get("portfolios", raw)
        loaded: Dict[str, Portfolio] = {}
        for k, v in raw_ports.items():
            loaded[str(k)] = Portfolio(
                name=v.get("name", f"Carteira {k}"),
                tickers=v.get("tickers", []),
                benchmark=v.get("benchmark", "^BVSP"),
            )
        # meta opcional
        meta = raw.get("_meta", {}) if isinstance(raw, dict) else {}
        if meta.get("logo_src"):
            st.session_state.logo_src = meta.get("logo_src")
        return loaded
    except Exception:
        return None

def _get_config_url_from_env_or_query() -> str | None:
    # 1) secrets
    try:
        if "CONFIG_URL" in st.secrets:
            return st.secrets["CONFIG_URL"]
    except Exception:
        pass
    # 2) query param (?config=URL)
    try:
        qp = st.query_params
        if "config" in qp:
            v = qp.get("config")
            return v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else None)
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            if "config" in qp:
                arr = qp.get("config")
                return arr[0] if isinstance(arr, list) and arr else None
        except Exception:
            pass
    return None
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        loaded: Dict[str, Portfolio] = {}
        for k, v in raw.items():
            loaded[str(k)] = Portfolio(
                name=v.get("name", f"Carteira {k}"),
                tickers=v.get("tickers", []),
                benchmark=v.get("benchmark", "^BVSP"),
            )
        return loaded
    except Exception:
        return None

@dataclass
class Portfolio:
    name: str
    tickers: List[Dict]  # [{"Ticker":"VIVT3", "Weight": 25.0}, ...]
    benchmark: str

    def to_dict(self):
        return asdict(self)

# Mapeia ticker bruto para Yahoo Finance (regra simples p/ B3)
def map_to_yahoo(t: str) -> str:
    if not t:
        return t
    t = t.strip().upper()
    if t.startswith("^"):  # índices do Yahoo
        return t
    if t.endswith(".SA"):
        return t
    # Heurística B3: termina com dígito (3/4/5/6/11/34, etc.)
    if re.search(r"\d$", t):
        return f"{t}.SA"
    return t  # US/europa, etc.

# Cache de preços
@st.cache_data(show_spinner=False, ttl=600)
def get_prices(tickers: List[str], start: date) -> pd.DataFrame:
    """Baixa preços com yfinance e retorna um DataFrame de closes (1 coluna por ticker).
    Observação: com auto_adjust=True, o yfinance não entrega 'Adj Close'; o 'Close' já vem ajustado.
    """
    # Remove vazios/duplicados e normaliza símbolos
    syms = sorted(set([map_to_yahoo(x) for x in tickers if str(x).strip()]))
    if not syms:
        return pd.DataFrame()

    df = yf.download(
        tickers=syms,
        start=start,
        progress=False,
        auto_adjust=True,  # 'Close' já ajustado; 'Adj Close' não existe neste modo
        group_by="ticker",
        threads=True,
        interval="1d",
    )

    # Normaliza para colunas simples (Close por ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Tenta usar 'Close'; se não houver, tenta 'Adj Close'; senão, usa a primeira coluna disponível
        lvl1 = df.columns.get_level_values(1)
        if "Close" in set(lvl1):
            closes = pd.concat({sym: df[sym]["Close"] for sym in syms if sym in df.columns.levels[0]}, axis=1)
        elif "Adj Close" in set(lvl1):
            closes = pd.concat({sym: df[sym]["Adj Close"] for sym in syms if sym in df.columns.levels[0]}, axis=1)
        else:
            first_metric = lvl1.unique().tolist()[0]
            closes = pd.concat({sym: df[sym][first_metric] for sym in syms if sym in df.columns.levels[0]}, axis=1)
    else:
        # Single ticker: colunas simples
        cols = df.columns.tolist()
        if "Close" in cols:
            closes = df[["Close"]].rename(columns={"Close": syms[0]})
        elif "Adj Close" in cols:
            closes = df[["Adj Close"]].rename(columns={"Adj Close": syms[0]})
        else:
            # fallback: pega a última coluna numérica
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return pd.DataFrame()
            closes = df[[num_cols[-1]]].rename(columns={num_cols[-1]: syms[0]})

    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="all")
    return closes

# Helpers de data

def month_start(dtref: date) -> date:
    return dtref.replace(day=1)

def year_start(dtref: date) -> date:
    return dtref.replace(month=1, day=1)

# Retorno entre primeira e última observação após uma data de início

def _period_return(series: pd.Series, start_date: date) -> float:
    if series is None or series.empty:
        return np.nan
    s = series.dropna()
    s = s[s.index.date >= start_date]
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1.0)

# Retorno diário (último fechamento vs penúltimo)

def _daily_return(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-2] - 1.0)

# Monta índice base-100 de uma carteira (retornos diários * pesos)

def build_port_index(price_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.copy().astype(float)
    if w.isna().all() or w.sum() == 0:
        w = pd.Series(np.repeat(1.0, len(w)), index=w.index)
    w = w / w.sum()
    rets = price_df.pct_change(fill_method=None).fillna(0.0)
    port_ret = (rets[w.index] * w).sum(axis=1)
    idx = (1.0 + port_ret).cumprod() * 100.0
    return idx

# Formatação %

def pct(x):
    return "-" if pd.isna(x) else f"{x*100:,.2f}%"

# Aceita pesos com vírgula ("12,5"), ponto ("12.5") ou com símbolo "%" e faz bound 0..100
def parse_weight(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace("%", "")
    if not s:
        return np.nan
    s = s.replace(" ", "")
    # Se possuir ponto e vírgula e a vírgula for o último separador, assume formato pt-BR (1.234,56)
    if "," in s and "." in s and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        val = float(s)
    except Exception:
        return np.nan
    # Limita 0..100 (%); mantém como porcentagem (não fração)
    return float(max(0.0, min(100.0, val)))

# ==========================
# Estado inicial (4 carteiras)
# ==========================
DEFAULT_PORTS = [
    Portfolio(name="Carteira 1", tickers=[{"Ticker": "VIVT3", "Weight": 25.0},
                                           {"Ticker": "TOTS3", "Weight": 25.0},
                                           {"Ticker": "EQTL3", "Weight": 25.0},
                                           {"Ticker": "EGIE3", "Weight": 25.0}], benchmark="^BVSP"),
    Portfolio(name="Carteira 2", tickers=[{"Ticker": "ELET3", "Weight": 50.0},
                                           {"Ticker": "CPLE6", "Weight": 50.0}], benchmark="^BVSP"),
    Portfolio(name="Carteira 3", tickers=[{"Ticker": "VIVT3", "Weight": 50.0},
                                           {"Ticker": "TIMP3", "Weight": 50.0}], benchmark="^BVSP"),
    Portfolio(name="Carteira 4", tickers=[{"Ticker": "IVVB11", "Weight": 100.0}], benchmark="^GSPC"),
]

if "portfolios" not in st.session_state:
    # 1) query/secrets -> URL remota
    config_env_url = _get_config_url_from_env_or_query()
    if config_env_url:
        remote = _fetch_remote_config(config_env_url)
        if remote:
            st.session_state.portfolios = remote
            try:
                max_id = max(map(int, list(remote.keys())))
                st.session_state.next_id = max_id + 1
            except Exception:
                st.session_state.next_id = len(remote) + 1
            st.session_state.config_url = config_env_url
        else:
            # 2) persistido local
            persisted = _load_persisted_portfolios()
            if persisted:
                st.session_state.portfolios = persisted
                try:
                    max_id = max(map(int, list(persisted.keys())))
                    st.session_state.next_id = max_id + 1
                except Exception:
                    st.session_state.next_id = len(persisted) + 1
            else:
                st.session_state.portfolios = {str(i+1): p for i, p in enumerate(DEFAULT_PORTS)}
                st.session_state.next_id = len(st.session_state.portfolios) + 1
    else:
        # 2) persistido local
        persisted = _load_persisted_portfolios()
        if persisted:
            st.session_state.portfolios = persisted
            try:
                max_id = max(map(int, list(persisted.keys())))
                st.session_state.next_id = max_id + 1
            except Exception:
                st.session_state.next_id = len(persisted) + 1
        else:
            st.session_state.portfolios = {str(i+1): p for i, p in enumerate(DEFAULT_PORTS)}
            st.session_state.next_id = len(st.session_state.portfolios) + 1


# ======================
# Sidebar – Configuração
# ======================
with st.sidebar:
    st.header("⚙️ Configurações")

    # Logo: caminho local no repo ou URL absoluta
    logo_src = st.text_input(
        "Logo (caminho ou URL)",
        value=st.session_state.get("logo_src", "assets/icone_completo_2022_fundo_titanio.png"),
        help="Ex.: assets/logo.png dentro do repo, ou uma URL (PNG/SVG/JPG).",
        key="logo_src_input",
    )
    st.session_state.logo_src = logo_src

    # Config remota (opcional)
    config_url = st.text_input(
        "Config remota (URL RAW do JSON)",
        value=st.session_state.get("config_url", ""),
        help="Cole aqui a URL RAW do JSON no GitHub/Gist. Use ?config=URL no link para autocarregar.",
        key="config_url_input",
    )
    st.session_state.config_url = config_url
    load_remote_btn = st.button("↻ Carregar desta URL", use_container_width=True)
    if load_remote_btn and config_url:
        remote = _fetch_remote_config(config_url)
        if remote:
            st.session_state.portfolios = remote
            try:
                max_id = max(map(int, list(remote.keys())))
                st.session_state.next_id = max_id + 1
            except Exception:
                st.session_state.next_id = len(remote) + 1
            _save_persisted_portfolios(st.session_state.portfolios)
            st.rerun()
        else:
            st.warning("Não foi possível carregar a URL fornecida.")

    st.markdown("---")
    # Período do gráfico (lookback)
    lookback = st.selectbox(
        "Período do gráfico",
        ["3M", "6M", "12M", "YTD", "24M", "MAX"],
        index=2,
        help="Controla a janela do gráfico de linhas (base 100).",
    )

    # Botão de atualização (limpa cache de preços)
    if st.button("🔄 Atualizar agora", use_container_width=True):
        st.cache_data.clear()
        st.toast("Dados atualizados.")

    st.markdown("---")

    # Adicionar/excluir carteiras
    # Entrada de nome customizado para nova carteira
    new_port_default = f"Carteira {st.session_state.next_id}"
    new_port_name = st.text_input("Nome da nova carteira", value=new_port_default, key="new_port_name")

    if st.button("➕ Adicionar carteira", use_container_width=True):
        new_id = str(st.session_state.next_id)
        name_to_use = st.session_state.get("new_port_name", new_port_default) or new_port_default
        st.session_state.portfolios[new_id] = Portfolio(
            name=name_to_use.strip(),
            tickers=[{"Ticker": "", "Weight": 0.0}],
            benchmark="^BVSP",
        )
        st.session_state.next_id += 1
        _save_persisted_portfolios(st.session_state.portfolios)
        st.rerun()

    del_id = st.text_input("ID para excluir", placeholder="ex.: 3")
    if st.button("🗑️ Excluir carteira", use_container_width=True) and del_id:
        if del_id in st.session_state.portfolios:
            st.session_state.portfolios.pop(del_id)
            _save_persisted_portfolios(st.session_state.portfolios)
            st.rerun()
        else:
            st.warning("ID não encontrado.")

    st.markdown("---")

    # Exportar / importar configuração
    if st.session_state.portfolios:
        export = {pid: p.to_dict() for pid, p in st.session_state.portfolios.items()}
        
    up = st.file_uploader("Carregar configuração JSON", type=["json"])
    if up is not None:
        try:
            data = json.load(up)
            new_state = {}
            for k, v in data.items():
                new_state[str(k)] = Portfolio(
                    name=v.get("name", f"Carteira {k}"),
                    tickers=v.get("tickers", []),
                    benchmark=v.get("benchmark", "^BVSP"),
                )
            st.session_state.portfolios = new_state
            # Ajusta next_id
            try:
                max_id = max(map(int, list(new_state.keys())))
                st.session_state.next_id = max_id + 1
            except Exception:
                st.session_state.next_id = len(new_state) + 1
            st.success("Config carregada.")
            _save_persisted_portfolios(st.session_state.portfolios)
        except Exception as e:
            st.error(f"Falha ao carregar JSON: {e}")

# ======================
# Main – Abas por carteira
# ======================

# Cabeçalho com logo + título
header_c1, header_c2 = st.columns([0.22, 1.78])
with header_c1:
    try:
        if "logo_bytes" in st.session_state and st.session_state.logo_bytes:
            st.image(st.session_state.logo_bytes, use_container_width=True)
        else:
            st.image(st.session_state.get("logo_src", "assets/icone_completo_2022_fundo_titanio.png"), use_container_width=True)
    except Exception:
        pass
with header_c2:
    st.title("📊 Monitor de Carteiras Recomendadas")
    st.caption("Edite tickers, pesos, benchmark e acompanhe variações (Dia, MTD, YTD, 12M) com gráfico base-100.")

if not st.session_state.portfolios:
    st.info("Nenhuma carteira. Adicione uma pela barra lateral.")
    st.stop()

# Ordena por ID
ordered = sorted(st.session_state.portfolios.items(), key=lambda x: int(x[0]))
labels = [f"#{pid} – {p.name}" for pid, p in ordered]

_tabs = st.tabs(labels)

# Janela mínima para obter 12M/YTD (baixa mais dados para robustez)
# Baixa histórico completo por padrão (melhor para preset MAX)
start_download = date(1990, 1, 1)

for (pid, portfolio), tab in zip(ordered, _tabs):
    with tab:
        colL, colR = st.columns([1.05, 1.4])
        with colL:
            st.subheader("Configuração")
            # Nome da carteira
            new_name = st.text_input("Nome", value=portfolio.name, key=f"name_{pid}")
            if new_name != portfolio.name:
                portfolio.name = new_name
                _save_persisted_portfolios(st.session_state.portfolios)

            # Benchmark
            bench_sel = st.selectbox(
                "Benchmark (presets)", list(BENCH_PRESETS.keys()), index=0, key=f"benchsel_{pid}",
            )
            bench_custom = st.text_input(
                "Ou digite um benchmark/manual (Yahoo)",
                value=portfolio.benchmark,
                key=f"benchcustom_{pid}",
                help="Ex.: ^BVSP, BOVA11.SA, ^GSPC, USDBRL=X, etc. Para ações da B3, use o ticker puro (ex.: VIVT3) que adicionamos .SA automaticamente.",
            )
            # Se usuário alterou manual, prioriza manual; senão usa preset
            bench_symbol = bench_custom.strip() if bench_custom.strip() else BENCH_PRESETS[bench_sel]
            old_bench = portfolio.benchmark
            if bench_symbol != old_bench:
                portfolio.benchmark = bench_symbol
                _save_persisted_portfolios(st.session_state.portfolios)

            st.markdown("")
            st.markdown("**Ativos & Pesos (%)**")
            st.caption("Dica: você pode digitar 12,5 ou 12.5; também aceita '12%'.")
            df_init = pd.DataFrame(portfolio.tickers or [{"Ticker": "", "Weight": 0.0}]).copy()
            # Força dtype 'object'/string p/ permitir TextColumn + vírgula
            if "Ticker" in df_init.columns:
                df_init["Ticker"] = df_init["Ticker"].astype(str)
            if "Weight" not in df_init.columns:
                df_init["Weight"] = ""
            else:
                df_init["Weight"] = df_init["Weight"].apply(lambda x: "" if pd.isna(x) else str(x))
            df_init["Weight"] = df_init["Weight"].astype(object)

            df_edit = st.data_editor(
                df_init,
                num_rows="dynamic",
                use_container_width=True,
                key=f"edit_{pid}",
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", help="Ex.: VIVT3, TOTS3, AAPL, BOVA11.SA"),
                    "Weight": st.column_config.TextColumn("Weight (%)", help="Use vírgula ou ponto; aceita 12,5, 12.5 ou 12%"),
                },
                hide_index=True,
            )
            # Persistência local
            tick_rows = (
                df_edit.assign(Ticker=lambda d: d["Ticker"].fillna("").astype(str).str.upper())
                      .assign(Weight=lambda d: d["Weight"].apply(parse_weight))
            )
            # Remove linhas vazias
            tick_rows = tick_rows[tick_rows["Ticker"].str.strip() != ""]
            portfolio.tickers = tick_rows.to_dict(orient="records")
            _save_persisted_portfolios(st.session_state.portfolios)

        with colR:
            st.subheader("Resultados")
            # Lista final de tickers e pesos
            tickers = [row["Ticker"] for row in portfolio.tickers]
            weights = [row["Weight"] for row in portfolio.tickers]
            if not tickers:
                st.info("Inclua ao menos um ticker.")
                continue
            # Mapeia para Yahoo e baixa preços
            all_syms = [map_to_yahoo(t) for t in tickers]
            bench_sym = map_to_yahoo(portfolio.benchmark)
            price_df = get_prices(all_syms + [bench_sym], start=start_download)
            if price_df.empty:
                st.warning("Sem dados retornados. Verifique os tickers/benchmark.")
                continue

            # Separa benchmark
            missing_syms = [s for s in all_syms if s not in price_df.columns]
            if bench_sym not in price_df.columns:
                st.warning("Benchmark sem dados – verifique o símbolo.")
                continue
            if missing_syms:
                st.warning(f"Sem dados para: {', '.join(missing_syms)}")
            used_syms = [s for s in all_syms if s in price_df.columns]
            if not used_syms:
                st.warning("Nenhum ticker válido com dados.")
                continue

            px_assets = price_df[used_syms]
            px_bench = price_df[bench_sym]

            # Pesos normalizados
            w = pd.Series(pd.to_numeric(weights, errors="coerce"), index=used_syms)
            if w.isna().all() or w.sum() == 0:
                w = pd.Series(np.repeat(1.0, len(used_syms)), index=used_syms)
            w = w / w.sum()

            today = date.today()
            start_12m = today - relativedelta(years=1)
            start_mtd = month_start(today)
            start_ytd = year_start(today)

            # Tabela de variações
            rows = []
            for sym in used_syms:
                s = px_assets[sym]
                rows.append({
                    "Ticker": sym,
                    "Dia": _daily_return(s),
                    "MTD": _period_return(s, start_mtd),
                    "YTD": _period_return(s, start_ytd),
                    "12M": _period_return(s, start_12m),
                })
            tbl = pd.DataFrame(rows)

            # Carteira (índice base 100)
            port_idx = build_port_index(px_assets, w)
            # Retornos carteira nas janelas
            port_row = {
                "Ticker": "CARTEIRA (wtd)",
                "Dia": _daily_return(port_idx),
                "MTD": _period_return(port_idx, start_mtd),
                "YTD": _period_return(port_idx, start_ytd),
                "12M": _period_return(port_idx, start_12m),
            }

            # Benchmark – montar índice base 100 para usar mesmas funções
            bench_idx = (px_bench / px_bench.dropna().iloc[0]) * 100.0
            bench_row = {
                "Ticker": f"Benchmark [{bench_sym}]",
                "Dia": _daily_return(px_bench),
                "MTD": _period_return(px_bench, start_mtd),
                "YTD": _period_return(px_bench, start_ytd),
                "12M": _period_return(px_bench, start_12m),
            }

            tbl = pd.concat([tbl, pd.DataFrame([port_row, bench_row])], ignore_index=True)

            # Formatação visual
            fmt_tbl = tbl.copy()
            for c in ["Dia", "MTD", "YTD", "12M"]:
                fmt_tbl[c] = fmt_tbl[c].apply(pct)

            # =====================
            # Gráfico – base 100
            # =====================
            st.subheader("Gráfico (base 100)")

            # Seleção de período: presets ou personalizado
            mode = st.radio("Período do gráfico", ["Presets", "Personalizado"], horizontal=True, key=f"mode_{pid}")

            if mode == "Presets":
                preset = st.selectbox("Presets", ["3M", "6M", "12M", "YTD", "24M", "MAX"], index=2, key=f"preset_{pid}")
                if preset == "3M":
                    start_g = today - relativedelta(months=3)
                    end_g = today
                elif preset == "6M":
                    start_g = today - relativedelta(months=6)
                    end_g = today
                elif preset == "12M":
                    start_g = today - relativedelta(years=1)
                    end_g = today
                elif preset == "24M":
                    start_g = today - relativedelta(years=2)
                    end_g = today
                elif preset == "YTD":
                    start_g = start_ytd
                    end_g = today
                else:
                    start_g = price_df.index.date.min() if len(price_df) else today - relativedelta(years=2)
                    end_g = price_df.index.date.max() if len(price_df) else today
            else:
                min_d = price_df.index.date.min()
                max_d = price_df.index.date.max()
                default_start = max(min_d, today - relativedelta(months=6))
                dr = st.date_input(
                    "Intervalo personalizado",
                    value=(default_start, max_d),
                    min_value=min_d,
                    max_value=max_d,
                    key=f"daterange_{pid}"
                )
                if isinstance(dr, tuple) and len(dr) == 2:
                    start_g, end_g = dr
                else:
                    start_g, end_g = default_start, max_d

            # Rebase 100 na janela
            def rebase_100(series: pd.Series, start_date: date, end_date: date | None = None) -> pd.Series:
                s = series.dropna()
                if end_date is None:
                    end_date = s.index.date.max() if len(s) else start_date
                mask = (s.index.date >= start_date) & (s.index.date <= end_date)
                s = s[mask]
                if s.empty:
                    return s
                return (s / s.iloc[0]) * 100.0

            port_g = rebase_100(port_idx, start_g, end_g)
            bench_g = rebase_100(px_bench, start_g, end_g)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=port_g.index, y=port_g, mode="lines", name=f"{portfolio.name}"))
            fig.add_trace(go.Scatter(x=bench_g.index, y=bench_g, mode="lines", name=f"Benchmark [{bench_sym}]"))

            with st.expander("Opções do gráfico"):
                show_components = st.checkbox("Mostrar componentes da carteira", value=False, key=f"showc_{pid}")
                if show_components:
                    for sym in used_syms:
                        comp_idx = rebase_100(px_assets[sym], start_g, end_g)
                        fig.add_trace(go.Scatter(x=comp_idx.index, y=comp_idx, mode="lines", name=sym, line=dict(width=1)))

            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="Base 100",
                xaxis_title="Data",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabela abaixo do gráfico
            st.subheader("Tabela de variações")
            st.dataframe(
                fmt_tbl,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Baixar tabela (CSV)",
                data=tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"variacoes_{portfolio.name.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{pid}_{abs(hash(portfolio.name))}",
            )

            st.markdown(
                "<span class='small'>Dica: para B3, digite o ticker sem sufixo (ex.: **VIVT3**) – adicionamos `.SA` automaticamente. Benchmarks aceitam índices do Yahoo (ex.: **^BVSP**, **^GSPC**) e moedas (ex.: **USDBRL=X**).</span>",
                unsafe_allow_html=True,
            )

# Fim
