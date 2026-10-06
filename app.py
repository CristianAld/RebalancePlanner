"""Rebalance Planner with Risk Check -- Streamlit UI only.

All math lives in drift.py, rebalance.py and risk.py.
    streamlit run app.py
"""

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from drift import compute_drift
from rebalance import corridor_trades, full_trades, turnover
from risk import (annualized_cov, load_prices, past_year, portfolio_vol, risk_contributions,
                  tracking_error)

DATA = Path(__file__).resolve().parent / "data"
MIN_TRADE = 0.01

INK = "#111111"
MUTED = "#6b6b68"
RANGE_FILL = "#e4e4e0"
BEFORE_COLOR = "#b5b5b0"   # the past recedes
AFTER_COLOR = "#111111"    # the proposal leads
BREACH_COLOR = "#d03b3b"   # status: critical

FUND_NAMES = {
    "VTI": "US stocks",
    "VXUS": "International stocks",
    "BND": "US bonds",
    "VNQ": "Real estate",
    "GLD": "Gold",
    "SHV": "Short-term Treasuries",
}

# Model mixes (target % per fund), lowest risk first. "Current plan" comes from the account file.
MODELS = {
    "Conservative": ({"VTI": 20, "VXUS": 10, "BND": 45, "VNQ": 5, "GLD": 5, "SHV": 15},
                     "Mostly bonds and cash."),
    "Balanced": ({"VTI": 30, "VXUS": 15, "BND": 35, "VNQ": 5, "GLD": 5, "SHV": 10},
                 "About half stocks, half bonds and cash."),
    "Growth": ({"VTI": 50, "VXUS": 25, "BND": 15, "VNQ": 5, "GLD": 5, "SHV": 0},
               "Mostly stocks, some bonds."),
    "Aggressive": ({"VTI": 60, "VXUS": 30, "BND": 0, "VNQ": 5, "GLD": 5, "SHV": 0},
                   "Almost all stocks."),
    "Equal weight": ({t: 100 / 6 for t in FUND_NAMES},
                     "Same share in every fund. A baseline, not a risk choice."),
}
CURRENT_PLAN = "Current plan"

CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top:2.25rem; max-width:1240px; }
[data-testid="stHeader"] { background:transparent; }
h2, h3 { letter-spacing:-.015em; }
.brand { display:flex; flex-wrap:wrap; align-items:baseline; gap:.25rem .75rem; margin:0 0 1rem; }
.brand b { font-size:1.05rem; font-weight:800; letter-spacing:-.01em; }
.brand span { color:#6b6b68; font-size:.9rem; }
.hero { background:#000; color:#fff; padding:1.6rem 1.75rem 1.5rem; margin:0 0 1rem; }
.hero .eyebrow { font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:#9a9a95; margin:0 0 .6rem; }
.hero h1 { color:#fff; font-size:clamp(1.45rem, 3.2vw, 2.1rem); line-height:1.15; font-weight:800; margin:0 0 .5rem; padding:0; letter-spacing:-.02em; }
.hero p { color:#cfcfca; font-size:.95rem; line-height:1.5; margin:0; font-variant-numeric:tabular-nums; }
[data-testid="stMetricLabel"] p { font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:#6b6b68; font-weight:600; }
[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
[data-testid="stTab"] p { font-size:.95rem; font-weight:600; }
.step { font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:#6b6b68; margin:1.5rem 0 -.6rem;
        border-top:2px solid #000; padding-top:.6rem; font-weight:600; }
.lede { font-size:.95rem; color:#111; margin:-.35rem 0 .5rem; font-variant-numeric:tabular-nums; }
.lede span { color:#6b6b68; }
.ticket { display:flex; flex-wrap:wrap; align-items:baseline; gap:.2rem .8rem; padding:.7rem 0;
          border-bottom:1px solid #ececea; }
.ticket .side { font-size:.66rem; font-weight:700; letter-spacing:.1em; padding:.15rem .5rem; min-width:3rem; text-align:center; }
.ticket .buy { background:#000; color:#fff; border:1.5px solid #000; }
.ticket .sell { background:#fff; color:#000; border:1.5px solid #000; }
.ticket .amt { font-size:1.05rem; font-weight:700; font-variant-numeric:tabular-nums; min-width:6.5rem; }
.ticket .fund { font-weight:600; }
.ticket .fund span { color:#6b6b68; font-weight:400; }
.ticket .why { flex-basis:100%; color:#6b6b68; font-size:.85rem; font-variant-numeric:tabular-nums; }
.hold { color:#6b6b68; font-size:.85rem; padding-top:.6rem; }
.note { font-size:.95rem; margin:.25rem 0 .25rem; }
</style>
"""

st.set_page_config(page_title="Rebalance Planner", page_icon="⚖️", layout="wide")
st.html(CSS)


@st.cache_data
def load_sample_account() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_account.csv")


@st.cache_data
def load_price_history() -> pd.DataFrame:
    return load_prices(DATA / "prices.csv")


@st.cache_data
def load_cov() -> pd.DataFrame:
    return annualized_cov(load_price_history())


def money(x: float, cents: bool = False) -> str:
    s = f"${abs(x):,.2f}" if cents else f"${abs(x):,.0f}"
    return f"-{s}" if x < -0.5 else s


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fund(t: str) -> str:
    return FUND_NAMES.get(t, "")


def say(text: str) -> None:
    """Markdown with literal dollar signs (Streamlit reads $...$ as LaTeX)."""
    st.markdown(text.replace("$", r"\$"))


def step(n: int, label: str) -> None:
    st.html(f'<div class="step">{n} · {label}</div>')


def lede(text: str) -> None:
    st.html(f'<p class="lede">{text}</p>')


def apply_edits(df: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Replay st.data_editor's pending edits so code above the table can use them."""
    df = df.copy()
    for row, changes in edits.get("edited_rows", {}).items():
        for col, val in changes.items():
            df.iat[int(row), df.columns.get_loc(col)] = val
    df = df.drop(df.index[edits.get("deleted_rows", [])])
    added = pd.DataFrame(edits.get("added_rows", []), columns=df.columns)
    return pd.concat([df, added], ignore_index=True) if len(added) else df


def clean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Editor output -> rows with a ticker, upper-cased, blanks in `col` as 0."""
    df = df.dropna(subset=["ticker"]).fillna({col: 0})
    df = df.assign(ticker=df["ticker"].astype(str).str.strip().str.upper())
    return df[df["ticker"] != ""]


def chart_layout(fig: go.Figure, height: int, **kw) -> None:
    fig.update_layout(
        height=height, margin=dict(l=0, r=10, t=10, b=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(family="Inter, sans-serif", color=INK),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK)),
    )
    fig.update_layout(**kw)


sample = load_sample_account()
model_mixes = {CURRENT_PLAN: (dict(zip(sample["ticker"], sample["target"] * 100)),
                              "The targets this account was set up with.")} | MODELS


def model_frame(name: str) -> pd.DataFrame:
    mix = model_mixes[name][0]
    return pd.DataFrame({"ticker": list(mix), "fund": [fund(t) for t in mix], "target": list(mix.values())})


# --- Account size <-> holdings, kept in sync -------------------------------
# The size dial and the holdings table both describe the account. Moving the dial
# rescales every holding (same mix, new dollars); editing a holding moves the dial.
ss = st.session_state
if "holdings_base" not in ss:
    ss.holdings_base = pd.DataFrame({
        "ticker": sample["ticker"], "fund": sample["ticker"].map(FUND_NAMES), "value": sample["value"],
    })
    ss.holdings_ver = 0
live = apply_edits(ss.holdings_base, ss.get(f"holdings_{ss.holdings_ver}", {}))
live_values = pd.to_numeric(live["value"], errors="coerce").fillna(0.0)
live_total = float(live_values.sum())
if "size" not in ss:
    ss.size = ss.last_size = live_total
elif ss.size != ss.last_size:                    # the dial moved
    if live_total > 0:
        live = live.assign(value=live_values * ss.size / live_total)
        live_total = ss.size
    ss.holdings_base = live.reset_index(drop=True)
    ss.holdings_ver += 1
    ss.last_size = ss.size
elif abs(live_total - ss.size) > 0.5:            # a holding was edited
    ss.size = ss.last_size = live_total

# --- Sidebar -------------------------------------------------------------------
st.sidebar.header("Account")
st.sidebar.number_input(
    "Account size ($)", min_value=0.0, step=25_000.0, format="%.0f", key="size",
    help="Every holding scales together, so the percentages, breaches and plan stay the same. "
         "Only the dollars change. Try $10,000 or $50,000,000.",
)

st.sidebar.header("Bands")
abs_band = st.sidebar.slider(
    "Any fund (± pts)", 1.0, 15.0, 5.0, 0.5, format="%.1f",
    help="How far any fund may drift, in percentage points of the account, before it's traded. "
         "Technical name: absolute band.",
) / 100
abs_example = st.sidebar.empty()
rel_band = st.sidebar.slider(
    "Small funds (± % of target)", 5, 100, 25, 5,
    help="Scales the allowance to the fund's own target, so a small fund can't double or vanish "
         "inside the first band. Technical name: relative band.",
) / 100
rel_example = st.sidebar.empty()
with st.sidebar.expander("How it works"):
    st.markdown(
        "1. **Choose a mix.** Start from a model and edit any target.\n"
        "2. **Measure drift.** Each fund's share of the account vs. its target.\n"
        "3. **Flag breaches.** A fund breaches if it breaks *either* band.\n"
        "4. **Trade only breaches.** They go back to target; leftover cash goes to in-range "
        "funds on the right side of target.\n"
        "5. **Check risk.** Volatility from a year of daily prices, before and after.\n\n"
        "**Wider bands:** fewer trades, more drift. **Tighter:** closer to plan, more trading."
    )

# --- Page frame: filled in once the inputs below are read ------------------------
st.html('<div class="brand"><b>Rebalance Planner</b>'
        "<span>The fewest trades that put a portfolio back on its target mix, with the risk impact.</span></div>")
summary = st.container()
tab_mix, tab_rebal = st.tabs(["Target mix", "Rebalance"])

with tab_mix:
    mix_left, mix_right = st.columns([5, 6], gap="large")
    with mix_left:
        st.subheader("Choose your mix")
        choice = st.selectbox("Model", list(model_mixes), key="model",
                              help="A starting point. Edit any target below and everything updates.")
        st.caption(model_mixes[choice][1])
        targets_in = st.data_editor(
            model_frame(choice),
            key=f"targets_{choice}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            disabled=["fund"],
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "fund": st.column_config.TextColumn("Fund"),
                "target": st.column_config.NumberColumn("Target", format="%.1f%%", min_value=0, max_value=100,
                                                        help="Share of the account you want in this fund."),
            },
        )
        target_note = st.empty()

with tab_rebal:
    with st.expander(f"Holdings · {money(live_total)} · {len(live.dropna(subset=['ticker']))} funds"):
        st.caption("What each fund is worth today. Edit a value and the account size follows.")
        holdings_in = st.data_editor(
            ss.holdings_base,
            key=f"holdings_{ss.holdings_ver}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            disabled=["fund"],
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "fund": st.column_config.TextColumn("Fund"),
                "value": st.column_config.NumberColumn("Market value", format="dollar", min_value=0),
            },
        )

account = clean(holdings_in, "value")
target_rows = clean(targets_in, "target")


def stop(message: str) -> None:
    summary.error(message)
    st.stop()


if account["ticker"].duplicated().any():
    stop("Each ticker can appear only once in your holdings.")
if target_rows["ticker"].duplicated().any():
    stop("Each ticker can appear only once in your targets.")
if account["value"].sum() <= 0:
    stop("Enter at least one holding with a market value.")
target_sum = target_rows["target"].sum() / 100
if abs(target_sum - 1) > 1e-6:
    gap = (1 - target_sum) * 100
    msg = f"Targets add up to {target_sum:.1%}. {'Add' if gap > 0 else 'Remove'} {abs(gap):.1f} pts."
    target_note.warning(msg)
    stop(msg)
target_note.caption("✓ Adds up to 100%")

# --- The math (all in pure modules) ----------------------------------------
holdings = account.set_index("ticker")["value"]
targets = target_rows.set_index("ticker")["target"] / 100
total = holdings.sum()

drift = compute_drift(holdings, targets, abs_band, rel_band)
corridor = corridor_trades(drift)
full = full_trades(drift)
t_corr, t_full = turnover(corridor, MIN_TRADE), turnover(full, MIN_TRADE)
breached = drift.index[drift["breach"]]
n_breach = len(breached)

# Sidebar examples, worked on the real account
big = targets.idxmax()
small = targets[targets > 0].idxmin()
tb, ts = targets[big], targets[small]
abs_example.caption(f"{big} at {tb:.0%} may sit {max(tb - abs_band, 0):.1%}–{tb + abs_band:.1%}")
rel_example.caption(f"{small} at {ts:.0%} may sit {ts * (1 - rel_band):.2%}–{ts * (1 + rel_band):.2%}")

prices = load_price_history()
cov = load_cov()
w_before = drift["current_weight"]
w_after = (drift["value"] + corridor) / total
w_full = (drift["value"] + full) / total
try:
    vol_before = portfolio_vol(w_before, cov)
    vol_after = portfolio_vol(w_after, cov)
    vol_full = portfolio_vol(w_full, cov)
    rc_before = risk_contributions(w_before, cov) / vol_before
    rc_after = risk_contributions(w_after, cov) / vol_after
    rc_target = risk_contributions(targets, cov) / vol_full
    te_before = tracking_error(w_before, targets, cov)
    te_after = tracking_error(w_after, targets, cov)
    history = past_year(targets, prices)
    model_stats = {
        name: (portfolio_vol(pd.Series(mix) / 100, cov), past_year(pd.Series(mix) / 100, prices))
        for name, (mix, _) in MODELS.items()
    }
    risk_error = None
except ValueError as e:
    risk_error = str(e)

# --- Summary: verdict + KPIs ----------------------------------------------------
if n_breach == 0:
    headline = "Every fund is in range. No trades needed."
    sub = f"Volatility {vol_before:.2%} · target mix {vol_full:.2%}" if risk_error is None else ""
else:
    n = t_corr["n_trades"]
    headline = f"{n} trade{'s' if n != 1 else ''} bring{'' if n != 1 else 's'} the account back in line."
    parts = [f"{money(corridor[corridor > 0].sum())} each way"]
    if risk_error is None:
        parts.append(f"volatility {vol_before:.2%} → {vol_after:.2%} (target {vol_full:.2%})")
    saved = t_full["dollars"] - t_corr["dollars"]
    if saved > 0.5:
        parts.append(f"{money(saved)} less than a full rebalance")
    sub = " · ".join(parts)

with summary:
    st.html(
        f'<div class="hero"><p class="eyebrow">{money(total)} account · {n_breach} of {len(drift)} '
        f"funds off target</p><h1>{escape(headline)}</h1><p>{escape(sub)}</p></div>"
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Account", money(total), border=True, height="stretch", help="Total market value of every holding.")
    k2.metric("Off target", f"{n_breach} of {len(drift)}", border=True, height="stretch",
              help="Funds outside their allowed range.")
    k3.metric("Trades", t_corr["n_trades"], border=True, height="stretch", help="Buys and sells in the plan.")
    k4.metric("Traded", money(t_corr["dollars"]), delta=f"{t_corr['dollars'] / total:.1%} of account",
              delta_color="off", delta_arrow="off", border=True, height="stretch",
              help="All buys plus all sells. As a share of the account it's the same at any account size.")
    if risk_error is None:
        k5.metric("Volatility", f"{vol_after:.1%}", delta=f"{(vol_after - vol_before) * 100:+.2f} pts",
                  delta_color="inverse", border=True, height="stretch",
                  help="How much the account typically swings in a year, after the trades. "
                       f"The target mix is built for {vol_full:.1%}.")
    else:
        k5.metric("Volatility", "n/a", border=True, height="stretch")

# --- Tab 1: the target mix, in dollars and in risk ------------------------------
with mix_right:
    st.subheader(f"{money(total)} by target")
    alloc = targets[targets > 0].sort_values()
    afig = go.Figure(go.Bar(
        x=alloc * total, y=list(alloc.index), orientation="h",
        marker=dict(color=INK, cornerradius=4),
        text=[f"{money(v * total)} · {v * 100:.1f}%" for v in alloc], textposition="auto",
        insidetextanchor="end", insidetextfont=dict(color="#ffffff", size=12),
        outsidetextfont=dict(color=INK, size=12), cliponaxis=False,
        customdata=[fund(t) for t in alloc.index],
        hovertemplate="%{y} · %{customdata}<br>%{x:$,.0f}<extra></extra>",
    ))
    chart_layout(
        afig, 80 + 44 * len(alloc),
        xaxis=dict(visible=False, range=[0, alloc.max() * total * 1.45]),
        yaxis=dict(tickvals=list(alloc.index),
                   ticktext=[f"<b>{t}</b> <span style='color:{MUTED}'>{escape(fund(t))}</span>"
                             for t in alloc.index]),
    )
    st.plotly_chart(afig, width="stretch", config={"displayModeBar": False})

with tab_mix:
    st.subheader("Risk profile")
    if risk_error is not None:
        st.warning(f"{risk_error}. Prices cover: {', '.join(cov.index)}.")
    else:
        window = f"{prices.index[0]:%b %Y}–{prices.index[-1]:%b %Y}"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Yearly swing", f"±{vol_full:.1%}", border=True, height="stretch",
                  help="Volatility of the target mix. In about two years out of three, the account ends "
                       "within this much of where it started.")
        m2.metric("In dollars", f"±{money(vol_full * total)}", border=True, height="stretch",
                  help="The same swing at your account size.")
        m3.metric("Last 12 months", f"{history['total_return']:+.1%}", border=True, height="stretch",
                  help=f"What this mix returned {window}, held steady. Not a forecast.")
        m4.metric("Worst drop", f"{history['max_drawdown']:.1%}",
                  delta=money(history["max_drawdown"] * total), delta_color="off", delta_arrow="off",
                  border=True, height="stretch", help=f"Biggest fall from a high to a later low, {window}.")

        ladder_col, notes_col = st.columns([6, 5], gap="large")
        with ladder_col:
            st.markdown("**Versus our models**")
            rows = [(name, v, h["total_return"], h["max_drawdown"]) for name, (v, h) in model_stats.items()]
            rows.append(("Your mix", vol_full, history["total_return"], history["max_drawdown"]))
            rows.sort(key=lambda r: r[1])
            names = [r[0] for r in rows]
            lfig = go.Figure()
            for mine, color, size in [(False, BEFORE_COLOR, 13), (True, INK, 17)]:
                pick = [r for r in rows if (r[0] == "Your mix") == mine]
                lfig.add_scatter(
                    x=[r[1] * 100 for r in pick], y=[r[0] for r in pick],
                    mode="markers+text", text=[f"{r[1]:.1%}" for r in pick], textposition="middle right",
                    textfont=dict(size=12, color=INK),
                    marker=dict(size=size, color=color, symbol="diamond" if mine else "circle",
                                line=dict(width=2, color="#ffffff")),
                    customdata=[(r[2] * 100, r[3] * 100) for r in pick],
                    hovertemplate="<b>%{y}</b><br>Swing ±%{x:.1f}% a year<br>Last 12 months %{customdata[0]:+.1f}%"
                                  "<br>Worst drop %{customdata[1]:.1f}%<extra></extra>",
                )
            lo, hi = min(r[1] for r in rows) * 100, max(r[1] for r in rows) * 100
            chart_layout(
                lfig, 70 + 46 * len(rows), showlegend=False,
                xaxis=dict(title="Yearly swing (volatility)", ticksuffix="%", gridcolor="#eeeeea",
                           zeroline=False, range=[max(lo - 1.5, 0), hi + 2]),
                yaxis=dict(categoryorder="array", categoryarray=names[::-1], tickvals=names[::-1],
                           ticktext=[f"<b>{n}</b>" if n == "Your mix" else n for n in names[::-1]]),
            )
            st.plotly_chart(lfig, width="stretch", config={"displayModeBar": False})

        with notes_col:
            st.markdown("**Insights**")
            notes = []
            ladder = sorted((v, n) for n, (v, _) in model_stats.items() if n != "Equal weight")
            near = [n for v, n in ladder if abs(v - vol_full) < 0.0015]
            below = [(v, n) for v, n in ladder if v < vol_full]
            above = [(v, n) for v, n in ladder if v > vol_full]
            if near:
                notes.append(f"Risk in line with **{near[0]}**.")
            elif below and above:
                notes.append(f"Risk between **{below[-1][1]}** ({below[-1][0]:.1%}) "
                             f"and **{above[0][1]}** ({above[0][0]:.1%}).")
            elif above:
                notes.append(f"Less risk than **{above[0][1]}** ({above[0][0]:.1%}), our lowest.")
            else:
                notes.append(f"More risk than **{below[-1][1]}** ({below[-1][0]:.1%}), our highest.")

            # Risk share / money share = each fund's marginal risk per dollar
            money_share = targets.reindex(rc_target.index, fill_value=0)
            per_dollar = (rc_target / money_share.where(money_share >= 0.03)).dropna().sort_values(ascending=False)
            for i, t in enumerate(per_dollar.index[:2]):
                if per_dollar[t] >= 1.3:
                    notes.append(f"**{t}**: {pct(money_share[t])} of money, **{pct(rc_target[t])} of risk**"
                                 + (". Most risk per dollar." if i == 0 else "."))
            calm = [t for t in rc_target.index if targets.get(t, 0) >= 0.03 and rc_target[t] < 0.5 * targets[t]]
            if calm:
                notes.append(f"**{', '.join(calm)}** steady the account: real money, little risk.")
            if targets.max() >= 0.5:
                notes.append(f"Over half in **{targets.idxmax()}**: its bad year is your bad year.")
            new = [t for t in targets.index if targets[t] > 0 and t not in holdings.index]
            if new:
                notes.append(f"Not held yet: **{', '.join(new)}**. The plan buys {'it' if len(new) == 1 else 'them'}.")
            gone = [t for t in holdings.index if (t not in targets.index or targets[t] == 0) and holdings[t] > 0]
            if gone:
                notes.append(f"No target for **{', '.join(gone)}**. The plan sells {'it' if len(gone) == 1 else 'them'}.")
            st.markdown("\n".join(f"- {n}" for n in notes))
            st.caption(f"Daily prices, {window}. Not a forecast.")


# --- Tab 2: rebalance -----------------------------------------------------------
def why(t: str) -> str:
    r = drift.loc[t]
    vs = f"{pct(r['current_weight'])} vs. {pct(r['target_weight'])} target"
    pts = abs(r["drift_pts"]) * 100
    side = "over" if r["drift_pts"] > 0 else "under"
    if r["breach"]:
        if r["target_weight"] == 0:
            return "Not in the target mix · sold in full"
        if r["breach_reason"] == "absolute":
            return f"{vs} · {pts:.1f} pts {side}, past ±{abs_band * 100:g} pt band"
        if r["breach_reason"] == "relative":
            return (f"{vs} · {abs(r['drift_rel']) * 100:.0f}% {side} its target, "
                    f"past ±{rel_band * 100:g}% band")
        return f"{vs} · past both bands"
    if corridor[t] > 0:
        return f"{vs} · in range, gets leftover cash"
    return f"{vs} · in range, trimmed to fund buys"


order = list(drift.index)
with tab_rebal:
    row1_left, row1_right = st.columns(2, gap="large")
    row2_left, row2_right = st.columns(2, gap="large")

# Step 1: drift
with row1_left:
    step(1, "Drift")
    st.subheader(f"{n_breach} of {len(drift)} funds out of range" if n_breach else "All funds in range")

    labels = [f"<b>{t}</b><br><span style='font-size:11px;color:{MUTED}'>{escape(fund(t))}</span>" for t in order]
    y = list(range(len(order)))
    inside = ~drift["breach"]
    lo_pts = (drift["band_lo"] - drift["target_weight"]) * 100
    hi_pts = (drift["band_hi"] - drift["target_weight"]) * 100
    drift_pts = drift["drift_pts"] * 100
    reach = max(drift_pts.abs().max(), hi_pts.max(), 1.0) * 1.2

    fig = go.Figure()
    fig.add_bar(
        name="Allowed range", y=y, base=lo_pts, x=hi_pts - lo_pts, orientation="h",
        marker=dict(color=RANGE_FILL, cornerradius=4), width=0.5,
        customdata=list(zip(drift["band_lo"] * 100, drift["band_hi"] * 100)),
        hovertemplate="Allowed %{customdata[0]:.2f}% to %{customdata[1]:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        name="Target", x=[0, 0], y=[-0.5, len(order) - 0.5], mode="lines",
        line=dict(color=INK, width=2), hoverinfo="skip",
    )
    for mask, name, color in [(inside, "In range", INK), (~inside, "Out of range", BREACH_COLOR)]:
        pts = drift_pts[mask]
        fig.add_scatter(
            name=name, y=[i for i, m in zip(y, mask) if m], x=pts,
            mode="markers+text", text=[f"{v:+.1f}" for v in pts],
            textposition="top center",
            textfont=dict(size=11, color=INK),
            marker=dict(size=14, color=color, line=dict(width=2, color="#ffffff")),
            customdata=list(zip(drift.loc[mask, "current_weight"] * 100, drift.loc[mask, "target_weight"] * 100,
                                drift.loc[mask, "breach_reason"].replace("", "inside both bands"))),
            hovertemplate="Now %{customdata[0]:.1f}% vs. target %{customdata[1]:.1f}%<br>%{customdata[2]}"
                          "<extra>" + name + "</extra>",
        )
    chart_layout(
        fig, 110 + 58 * len(order),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12)),
        xaxis=dict(title="Points from target", range=[-reach, reach],
                   tickformat="+.0f", gridcolor="#eeeeea", zeroline=False),
        yaxis=dict(tickvals=y, ticktext=labels, range=[len(order) - 0.5, -0.5], showgrid=False),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with st.expander("Numbers"):
        st.dataframe(
            pd.DataFrame({
                "Fund": [fund(t) for t in order],
                "Now": drift["current_weight"] * 100,
                "Target": drift["target_weight"] * 100,
                "Allowed from": drift["band_lo"] * 100,
                "Allowed to": drift["band_hi"] * 100,
                "Drift (pts)": drift["drift_pts"] * 100,
                "Drift (% of target)": drift["drift_rel"] * 100,
                "Breach": drift["breach_reason"].replace("", "—"),
            }),
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.2f%%")
                for c in ["Now", "Target", "Allowed from", "Allowed to"]
            } | {
                "Drift (pts)": st.column_config.NumberColumn(format="%+.2f"),
                "Drift (% of target)": st.column_config.NumberColumn(format="%+.0f%%"),
            },
        )

# Step 2: trades
with row1_right:
    step(2, "Trades")
    if n_breach == 0:
        st.subheader("Nothing to trade")
        st.caption("Tighten the bands in the sidebar to see what would trade.")
    else:
        st.subheader(f"{t_corr['n_trades']} trades, net $0".replace("$", r"\$"),
                     help="Out-of-range funds go back to target. The cash that frees up (or uses) is "
                          "shared among in-range funds on the right side of their target, so nothing "
                          "moves further from where it should be.")
        tickets = []
        for t in sorted(order, key=lambda t: -abs(corridor[t])):
            amt = corridor[t]
            if abs(amt) < MIN_TRADE:
                continue
            side = "buy" if amt > 0 else "sell"
            tickets.append(
                f'<div class="ticket"><span class="side {side}">{side.upper()}</span>'
                f'<span class="amt">{money(abs(amt))}</span>'
                f'<span class="fund">{escape(t)} <span>{escape(fund(t))}</span></span>'
                f'<span class="why">{escape(why(t))}</span></div>'
            )
        held = [t for t in order if abs(corridor[t]) < MIN_TRADE]
        if held:
            tickets.append(f'<div class="hold">No trade: {", ".join(escape(t) for t in held)} (in range)</div>')
        st.html("".join(tickets))

# Step 3: risk
with row2_left:
    step(3, "Risk")
    if risk_error is not None:
        st.subheader("Risk check unavailable")
        st.warning(f"{risk_error}. Prices cover: {', '.join(cov.index)}.")
    else:
        if t_corr["n_trades"]:
            st.subheader(f"Volatility {vol_before:.1%} → {vol_after:.1%}")
            lede(f"Target mix {vol_full:.2%} <span>·</span> yearly swing "
                 f"±{money(vol_before * total)} → ±{money(vol_after * total)}")
        else:
            st.subheader(f"Volatility {vol_before:.1%}, unchanged")
            lede(f"Target mix {vol_full:.2%} <span>·</span> yearly swing ±{money(vol_before * total)}")

        top_risk = rc_before.idxmax()
        moved = abs(rc_after[top_risk] - rc_before[top_risk]) > 0.0005
        say(f"**{top_risk}**: {pct(w_before[top_risk])} of money, **{pct(rc_before[top_risk])} of risk**"
            + (f" → {pct(rc_after[top_risk])} after trades." if moved else "."))

        risk_tickers = [t for t in order if t in cov.index]
        rfig = go.Figure()
        for name, rc, color in [("Before", rc_before, BEFORE_COLOR), ("After trades", rc_after, AFTER_COLOR)]:
            rfig.add_bar(
                name=name, y=risk_tickers, x=rc[risk_tickers] * 100, orientation="h",
                marker=dict(color=color, cornerradius=4),
                hovertemplate="%{y}: %{x:.1f}% of risk<extra>" + name + "</extra>",
            )
        chart_layout(
            rfig, 80 + 52 * len(risk_tickers),
            barmode="group", bargap=0.35, bargroupgap=0.1, margin=dict(l=0, r=10, t=30, b=0),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
            xaxis=dict(title="Share of portfolio risk", ticksuffix="%", gridcolor="#eeeeea", zeroline=False),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(rfig, width="stretch", config={"displayModeBar": False})

        with st.expander("Numbers and method"):
            st.markdown("**Volatility** is how much the account typically moves in a year: about two years in "
                        "three, it ends within that range. The aim is the risk your target mix is built for, "
                        "not the lowest risk.")
            st.dataframe(
                pd.DataFrame({
                    "Fund": [fund(t) for t in risk_tickers],
                    "Money (before)": w_before[risk_tickers] * 100,
                    "Risk (before)": rc_before[risk_tickers] * 100,
                    "Money (after)": w_after[risk_tickers] * 100,
                    "Risk (after)": rc_after[risk_tickers] * 100,
                }, index=pd.Index(risk_tickers, name="ticker")),
                width="stretch",
                column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in [
                    "Money (before)", "Risk (before)", "Money (after)", "Risk (after)"]},
            )
            st.caption("One year of daily adjusted closes, annualized ×252. "
                       "Risk share is wᵢ(Σw)ᵢ / σ; shares add up to 100%.")

# Step 4: corridor vs. full
with row2_right:
    step(4, "Corridor vs. full rebalance")
    fewer_trades = t_full["n_trades"] - t_corr["n_trades"]
    closed = None
    if risk_error is None and te_before > 1e-9 and n_breach and fewer_trades:
        # Tracking error is never negative and is 0 after a full rebalance, so this stays within 0-100%.
        closed = 1 - te_after / te_before

    if n_breach == 0:
        st.subheader("No trades vs. " + f"{t_full['n_trades']} for a full rebalance")
    elif n_breach == len(drift):
        st.subheader("Same plan: every fund breaches")
    elif fewer_trades == 0:
        st.subheader("Same trades, either way")
    elif closed is not None and closed > 0:
        closed_txt = "Over 99%" if closed > 0.995 and te_after > 1e-9 else f"{closed:.0%}"
        st.subheader(f"Closes {closed_txt.lower() if closed_txt[0] == 'O' else closed_txt} of the gap",
                     help="The gap is how far the account is from the target mix, in risk terms. "
                          "A full rebalance closes 100%.")
    else:
        st.subheader(f"{fewer_trades} fewer trade{'s' if fewer_trades != 1 else ''}")

    gap = (lambda te: f" · gap {te:.2%}") if risk_error is None else (lambda te: "")
    c1, c2 = st.columns(2)
    with c1.container(border=True, height="stretch"):
        st.metric("Corridor · this plan", money(t_corr["dollars"]),
                  delta=f"{money(t_corr['dollars'] - t_full['dollars'])} vs. full", delta_color="inverse")
        st.caption(f"{t_corr['n_trades']} trades" + gap(te_after))
    with c2.container(border=True, height="stretch"):
        st.metric("Full rebalance", money(t_full["dollars"]))
        st.caption(f"{t_full['n_trades']} trades" + gap(0.0))

    if n_breach == len(drift):
        st.caption("Widen the bands to see the difference.")
    elif n_breach and fewer_trades == 0:
        st.caption(f"Fixing the {n_breach} breaches frees enough cash that every in-range fund gets a share.")

    with st.expander("Why trade less?"):
        st.markdown(
            "A full rebalance pushes *every* fund back to target, even ones barely off. The corridor plan "
            "leaves small drifts alone. Every trade costs something (spreads, commissions, taxes in a "
            "taxable account), so skipping trades that change little saves money.\n\n"
            "**The gap** is tracking error: how much the account's yearly return could differ from the "
            "target mix's. It's 0% after a full rebalance, so the share closed can never pass 100%."
            + (f" Today {te_before:.2%}; after this plan {te_after:.2%}." if risk_error is None else "")
        )
