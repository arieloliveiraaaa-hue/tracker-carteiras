# Streamlit – Monitor de Carteiras Recomendadas (B3 + Global)
# Deploy: suba este arquivo no GitHub e conecte em streamlit.app
# Requisitos (requirements.txt):
# streamlit
# yfinance
# pandas
# numpy
# plotly
# python-dateutil
# pytz
# requests

import json
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

# =============================
# Config da página
# =============================
st.set_page_config(page_title="Monitor de Carteiras", page_icon="📈", layout="wide")

# =============================
# Constantes e Presets
# =============================
DATE_TZ = "America/Sao_Paulo"
LOGO_DEFAULT = "assets/icone_completo_2022_fundo_titanio.png"
STATE_PATH = Path.home() / ".streamlit_carteiras_config.json"

BENCH_PRESETS = {
    "Ibovespa (^BVSP)": "^BVSP",
    "BOVA11 (ETF)": "BOVA11.SA",
    "S&P 500 (^GSPC)": "^GSPC",
    "NASDAQ 100 (^NDX)": "^NDX",
    "Dólar/Real (USDBRL)": "USDBRL=X",
}

# =============================
# Modelos e utilidades
# =============================
@dataclass
class Portfolio:
    name: str
    tickers: List[Dict]  # [{"Ticker": "VIVT3", "Weight": 25.0}, ...]
    benchmark: str

    def to_dict(self) -> Dict:
        return asdict(self)


def map_to_yahoo(t: str) -> str:
    """Mapeia ticker para o formato do Yahoo. Para B3, adiciona .SA se faltar."""
    if not t:
        return t
    t = t.strip().upper()
    if t.startswith("^"):
        return t
    if t.endswith(".SA"):
        return t
    if re.search("[0-9]$", t):  # termina com dígito (ex.: 3/4/5/6/11/34)
        return f"{t}.SA"
    return t


def pct(x: Optional[float]) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:,.2f}%"


def parse_weight(x) -> float:
    """Aceita "12,5", "12.5", "12%", "1.234,56"; devolve 0..100 (float)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace("%", "").replace(" ", "")
    if not s:
        return np.nan
    if "," in s and "." in s and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        val = float(s)
    except Exception:
        return np.nan
    return float(max(0.0, min(100.0, val)))


@st.cache_data(show_spinner=False, ttl=600)
def get_prices(tickers: List[str], start: date) -> pd.DataFrame:
    """Baixa preços diários ajustados (coluna Close) via yfinance.
    Retorna DataFrame com 1 coluna por símbolo.
    """
    syms = sorted(set([map_to_yahoo(x) for x in tickers if str(x).strip()]))
    if not syms:
        return pd.DataFrame()
    df = yf.download(
        tickers=syms,
        start=start,
        progress=False,
        auto_adjust=True,  # Close já ajustado
        group_by="ticker",
        threads=True,
        interval="1d",
    )
    # Normaliza para DataFrame simples de closes
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        closes = []
        for sym in syms:
            if sym in lvl0:
                if "Close" in set(lvl1):
                    closes.append(df[sym]["Close"].rename(sym))
                elif "Adj Close" in set(lvl1):
                    closes.append(df[sym]["Adj Close"].rename(sym))
        if not closes:
            return pd.DataFrame()
        closes = pd.concat(closes, axis=1)
    else:
        # single ticker
        cols = df.columns.tolist()
        if "Close" in cols:
            closes = df[["Close"]].rename(columns={"Close": syms[0]})
        elif "Adj Close" in cols:
            closes = df[["Adj Close"]].rename(columns={"Adj Close": syms[0]})
        else:
            # fallback: última col numérica
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return pd.DataFrame()
            closes = df[[num_cols[-1]]].rename(columns={num_cols[-1]: syms[0]})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="all")
    return closes


def _period_return(series: pd.Series, start_date: date) -> float:
    s = (series or pd.Series(dtype=float)).dropna()
    s = s[s.index.date >= start_date]
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def _daily_return(series: pd.Series) -> float:
    s = (series or pd.Series(dtype=float)).dropna()
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-2] - 1.0)


def build_port_index(price_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = (weights or pd.Series(dtype=float)).astype(float)
    if w.isna().all() or w.sum() == 0:
        w = pd.Series(np.repeat(1.0, len(price_df.columns)), index=price_df.columns)
    w = w / w.sum()
    rets = price_df.pct_change(fill_method=None).fillna(0.0)
    port_ret = (rets[w.index] * w).sum(axis=1)
    idx = (1.0 + port_ret).cumprod() * 100.0
    return idx

# =============================
# Persistência local (sem linhas soltas!)
# =============================

def _save_persisted_portfolios(portfolios_state: Dict[str, Portfolio]) -> None:
    try:
        data = {pid: p.to_dict() for pid, p in portfolios_state.items()}
        payload = {
            "portfolios": data,
            "_meta": {
                "logo_src": st.session_state.get("logo_src", LOGO_DEFAULT),
                "config_url": st.session_state.get("config_url", ""),
            },
        }
        STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_persisted_portfolios() -> Optional[Dict[str, Portfolio]]:
    try:
        if not STATE_PATH.exists():
            return None
        raw_text = STATE_PATH.read_text(encoding="utf-8")
        if not raw_text.strip():
            return None
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and "portfolios" in obj:
            ports = obj.get("portfolios", {})
            meta = obj.get("_meta", {})
            if isinstance(meta, dict):
                st.session_state["logo_src"] = meta.get("logo_src", LOGO_DEFAULT)
                st.session_state["config_url"] = meta.get("config_url", "")
        else:
            ports = obj
        loaded: Dict[str, Portfolio] = {}
        for k, v in (ports or {}).items():
            loaded[str(k)] = Portfolio(
                name=v.get("name", f"Carteira {k}"),
                tickers=v.get("tickers", []),
                benchmark=v.get("benchmark", "^BVSP"),
            )
        return loaded
    except Exception:
        return None

# =============================
# Config remota (opcional)
# =============================
try:
    import requests  # para config remota
except Exception:
    requests = None


def _fetch_remote_config(url: str) -> Optional[Dict[str, Portfolio]]:
    if not url or requests is None:
        return None
    try:
        r = requests.get(url, timeout=10)
        if not r.ok:
            return None
        obj = r.json()
        ports_raw = obj.get("portfolios", obj)
        meta = obj.get("_meta", {}) if isinstance(obj, dict) else {}
        if isinstance(meta, dict) and meta.get("logo_src"):
            st.session_state["logo_src"] = meta.get("logo_src")
        loaded: Dict[str, Portfolio] = {}
        for k, v in ports_raw.items():
            loaded[str(k)] = Portfolio(
                name=v.get("name", f"Carteira {k}"),
                tickers=v.get("tickers", []),
                benchmark=v.get("benchmark", "^BVSP"),
            )
        return loaded
    except Exception:
        return None


def _get_config_url_from_env_or_query() -> Optional[str]:
    # secrets
    try:
        if "CONFIG_URL" in st.secrets:
            return st.secrets["CONFIG_URL"]
    except Exception:
        pass
    # query param (?config=...)
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

# =============================
# Estado inicial
# =============================
DEFAULT_PORTS = [
    Portfolio(name="Carteira 1", tickers=[{"Ticker": "VIVT3", "Weight": 25.0}, {"Ticker": "TOTS3", "Weight": 25.0}, {"Ticker": "EQTL3", "Weight": 25.0}, {"Ticker": "EGIE3", "Weight": 25.0}], benchmark="^BVSP"),
    Portfolio(name="Carteira 2", tickers=[{"Ticker": "ELET3", "Weight": 50.0}, {"Ticker": "CPLE6", "Weight": 50.0}], benchmark="^BVSP"),
    Portfolio(name="Carteira 3", tickers=[{"Ticker": "VIVT3", "Weight": 50.0}, {"Ticker": "TIMP3", "Weight": 50.0}], benchmark="^BVSP"),
    Portfolio(name="Carteira 4", tickers=[{"Ticker": "IVVB11", "Weight": 100.0}], benchmark="^GSPC"),
]

if "portfolios" not in st.session_state:
    # tenta config remota
    cfg_url = _get_config_url_from_env_or_query()
    if cfg_url:
        remote = _fetch_remote_config(cfg_url)
        if remote:
            st.session_state.portfolios = remote
            st.session_state.config_url = cfg_url
            try:
                max_id = max(map(int, list(remote.keys())))
                st.session_state.next_id = max_id + 1
            except Exception:
                st.session_state.next_id = len(remote) + 1
        else:
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

# =============================
# Sidebar – Config
# =============================
with st.sidebar:
    st.header("⚙️ Configurações")
    # logo
    logo_src = st.text_input("Logo (caminho ou URL)", value=st.session_state.get("logo_src", LOGO_DEFAULT), help="Ex.: assets/logo.png ou URL pública")
    st.session_state.logo_src = logo_src

    # config remota
    config_url = st.text_input("Config remota (URL RAW JSON)", value=st.session_state.get("config_url", ""), help="Use GitHub RAW/Gist. Também funciona ?config=URL e Secrets.CONFIG_URL")
    st.session_state.config_url = config_url
    if st.button("↻ Carregar desta URL") and config_url:
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
    # adicionar carteira
    new_port_default = f"Carteira {st.session_state.get('next_id', 1)}"
    new_port_name = st.text_input("Nome da nova carteira", value=new_port_default)
    if st.button("➕ Adicionar carteira"):
        new_id = str(st.session_state.next_id)
        st.session_state.portfolios[new_id] = Portfolio(
            name=(new_port_name or new_port_default).strip(),
            tickers=[{"Ticker": "", "Weight": 0.0}],
            benchmark="^BVSP",
        )
        st.session_state.next_id += 1
        _save_persisted_portfolios(st.session_state.portfolios)
        st.rerun()

    if st.session_state.portfolios:
        del_id = st.selectbox("Excluir carteira (ID)", options=[""] + sorted(st.session_state.portfolios.keys(), key=lambda x: int(x)))
        if st.button("🗑️ Excluir carteira") and del_id:
            st.session_state.portfolios.pop(del_id, None)
            _save_persisted_portfolios(st.session_state.portfolios)
            st.rerun()

    st.markdown("---")
    # export/import
    export = {pid: p.to_dict() for pid, p in st.session_state.portfolios.items()}
    st.download_button("💾 Baixar configuração (JSON)", data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"), file_name="carteiras_config.json", mime="application/json")
    up = st.file_uploader("Carregar configuração JSON", type=["json"])
    if up is not None:
        try:
            data = json.load(up)
            new_state: Dict[str, Portfolio] = {}
            raw_ports = data.get("portfolios", data)
            for k, v in raw_ports.items():
                new_state[str(k)] = Portfolio(
                    name=v.get("name", f"Carteira {k}"),
                    tickers=v.get("tickers", []),
                    benchmark=v.get("benchmark", "^BVSP"),
                )
            st.session_state.portfolios = new_state
            try:
                max_id = max(map(int, list(new_state.keys())))
                st.session_state.next_id = max_id + 1
            except Exception:
                st.session_state.next_id = len(new_state) + 1
            _save_persisted_portfolios(st.session_state.portfolios)
            st.success("Config carregada.")
            st.rerun()
        except Exception as e:
            st.error(f"Falha ao carregar JSON: {e}")

# =============================
# Header com logo
# =============================
col_logo, col_title = st.columns([0.22, 1.78])
with col_logo:
    try:
        st.image(st.session_state.get("logo_src", LOGO_DEFAULT), use_container_width=True)
    except Exception:
        pass
with col_title:
    st.title("📊 Monitor de Carteiras Recomendadas")
    st.caption("Edite tickers, pesos, benchmark e acompanhe variações (Dia, MTD, YTD, 12M) com gráfico base-100.")

# =============================
# Abas de carteiras
# =============================
if not st.session_state.portfolios:
    st.info("Nenhuma carteira. Adicione uma pela barra lateral.")
    st.stop()

ordered = sorted(st.session_state.portfolios.items(), key=lambda x: int(x[0]))
labels = [f"#{pid} – {p.name}" for pid, p in ordered]
_tabs = st.tabs(labels)

# histórico amplo para suportar MAX de verdade
start_download = date(1990, 1, 1)

for (pid, portfolio), tab in zip(ordered, _tabs):
    with tab:
        colL, colR = st.columns([1.05, 1.4])
        with colL:
            st.subheader("Configuração")
            new_name = st.text_input("Nome", value=portfolio.name, key=f"name_{pid}")
            if new_name != portfolio.name:
                portfolio.name = new_name
                _save_persisted_portfolios(st.session_state.portfolios)

            bench_sel = st.selectbox("Benchmark (presets)", list(BENCH_PRESETS.keys()), index=0, key=f"benchsel_{pid}")
            bench_custom = st.text_input("Ou digite um benchmark/manual (Yahoo)", value=portfolio.benchmark, key=f"benchcustom_{pid}", help="Ex.: ^BVSP, BOVA11.SA, ^GSPC, USDBRL=X")
            bench_symbol = bench_custom.strip() if bench_custom.strip() else BENCH_PRESETS[bench_sel]
            if bench_symbol != portfolio.benchmark:
                portfolio.benchmark = bench_symbol
                _save_persisted_portfolios(st.session_state.portfolios)

            st.markdown("**Ativos & Pesos (%)**")
            st.caption("Dica: você pode digitar 12,5 ou 12.5; também aceita '12%'.")
            df_init = pd.DataFrame(portfolio.tickers or [{"Ticker": "", "Weight": 0.0}]).copy()
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
                key=f"edit_{pid}",
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", help="Ex.: VIVT3, TOTS3, AAPL, BOVA11.SA"),
                    "Weight": st.column_config.TextColumn("Weight (%)", help="Use vírgula ou ponto; aceita 12,5, 12.5 ou 12%"),
                },
            )
            tick_rows = (
                df_edit.assign(Ticker=lambda d: d["Ticker"].fillna("").astype(str).str.upper())
                      .assign(Weight=lambda d: d["Weight"].apply(parse_weight))
            )
            tick_rows = tick_rows[tick_rows["Ticker"].str.strip() != ""]
            portfolio.tickers = tick_rows.to_dict(orient="records")
            _save_persisted_portfolios(st.session_state.portfolios)

        with colR:
            st.subheader("Resultados")
            tickers = [row["Ticker"] for row in portfolio.tickers]
            weights = [row["Weight"] for row in portfolio.tickers]
            if not tickers:
                st.info("Inclua ao menos um ticker.")
                continue

            all_syms = [map_to_yahoo(t) for t in tickers]
            bench_sym = map_to_yahoo(portfolio.benchmark)
            price_df = get_prices(all_syms + [bench_sym], start=start_download)
            if price_df.empty:
                st.warning("Sem dados retornados. Verifique os tickers/benchmark.")
                continue

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

            w = pd.Series(pd.to_numeric(weights, errors="coerce"), index=used_syms)
            if w.isna().all() or w.sum() == 0:
                w = pd.Series(np.repeat(1.0, len(used_syms)), index=used_syms)
            w = w / w.sum()

            today = date.today()
            start_12m = today - relativedelta(years=1)
            start_mtd = today.replace(day=1)
            start_ytd = today.replace(month=1, day=1)

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

            port_idx = build_port_index(px_assets, w)
            port_row = {
                "Ticker": "CARTEIRA (wtd)",
                "Dia": _daily_return(port_idx),
                "MTD": _period_return(port_idx, start_mtd),
                "YTD": _period_return(port_idx, start_ytd),
                "12M": _period_return(port_idx, start_12m),
            }
            bench_row = {
                "Ticker": f"Benchmark [{bench_sym}]",
                "Dia": _daily_return(px_bench),
                "MTD": _period_return(px_bench, start_mtd),
                "YTD": _period_return(px_bench, start_ytd),
                "12M": _period_return(px_bench, start_12m),
            }
            tbl = pd.concat([tbl, pd.DataFrame([port_row, bench_row])], ignore_index=True)

            fmt_tbl = tbl.copy()
            for c in ["Dia", "MTD", "YTD", "12M"]:
                fmt_tbl[c] = fmt_tbl[c].apply(pct)

            # ===================== Gráfico (topo) =====================
            st.subheader("Gráfico (base 100)")
            mode = st.radio("Período do gráfico", ["Presets", "Personalizado"], horizontal=True, key=f"mode_{pid}")
            if mode == "Presets":
                preset = st.selectbox("Presets", ["3M", "6M", "12M", "YTD", "24M", "MAX"], index=2, key=f"preset_{pid}")
                if preset == "3M":
                    start_g = today - relativedelta(months=3); end_g = today
                elif preset == "6M":
                    start_g = today - relativedelta(months=6); end_g = today
                elif preset == "12M":
                    start_g = today - relativedelta(years=1); end_g = today
                elif preset == "24M":
                    start_g = today - relativedelta(years=2); end_g = today
                elif preset == "YTD":
                    start_g = start_ytd; end_g = today
                else:  # MAX = interseção real
                    starts, ends = [], []
                    for sym in used_syms:
                        s = px_assets[sym].dropna()
                        if not s.empty:
                            starts.append(s.index[0].date()); ends.append(s.index[-1].date())
                    sb = px_bench.dropna()
                    if not sb.empty:
                        starts.append(sb.index[0].date()); ends.append(sb.index[-1].date())
                    if starts and ends:
                        start_g = max(starts); end_g = min(ends)
                    else:
                        start_g = price_df.index.date.min() if len(price_df) else today
                        end_g = price_df.index.date.max() if len(price_df) else today
            else:
                min_d = price_df.index.date.min(); max_d = price_df.index.date.max()
                default_start = max(min_d, today - relativedelta(months=6))
                dr = st.date_input("Intervalo personalizado", value=(default_start, max_d), min_value=min_d, max_value=max_d, key=f"daterange_{pid}")
                if isinstance(dr, tuple) and len(dr) == 2:
                    start_g, end_g = dr
                else:
                    start_g, end_g = default_start, max_d

            def rebase_100(series: pd.Series, start_date: date, end_date: Optional[date] = None) -> pd.Series:
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
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"), yaxis_title="Base 100", xaxis_title="Data")
            st.plotly_chart(fig, use_container_width=True)

            # ===================== Tabela (abaixo) =====================
            st.subheader("Tabela de variações")
            st.dataframe(fmt_tbl, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇️ Baixar tabela (CSV)",
                data=tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"variacoes_{portfolio.name.replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"dl_{pid}_{abs(hash(portfolio.name))}",
            )
