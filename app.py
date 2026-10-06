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
from risk import annualized_cov, load_prices, portfolio_vol, risk_contributions

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

CSS = """
<style>
.hero { background:#000; color:#fff; padding:2rem 1.75rem 1.75rem; margin:0 0 1.25rem; }
.hero .eyebrow { font-size:.75rem; letter-spacing:.14em; text-transform:uppercase; color:#b5b5b0; margin:0 0 .75rem; }
.hero h1 { color:#fff; font-size:clamp(1.6rem, 4.5vw, 2.4rem); line-height:1.15; font-weight:800; margin:0 0 .75rem; padding:0; }
.hero p { color:#dcdcd8; font-size:1.02rem; line-height:1.5; margin:0; max-width:46rem; }
.step { font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#6b6b68; margin:1.75rem 0 -.5rem;
        border-top:3px solid #000; padding-top:.75rem; font-weight:600; }
.ticket { display:flex; flex-wrap:wrap; align-items:baseline; gap:.35rem .9rem; padding:.85rem 0;
          border-bottom:1px solid #e4e4e0; }
.ticket .side { font-size:.72rem; font-weight:700; letter-spacing:.1em; padding:.2rem .55rem; min-width:3.2rem; text-align:center; }
.ticket .buy { background:#000; color:#fff; border:1.5px solid #000; }
.ticket .sell { background:#fff; color:#000; border:1.5px solid #000; }
.ticket .amt { font-size:1.15rem; font-weight:700; font-variant-numeric:tabular-nums; min-width:6.5rem; }
.ticket .fund { font-weight:600; }
.ticket .fund span { color:#6b6b68; font-weight:400; }
.ticket .why { flex-basis:100%; color:#4a4a47; font-size:.92rem; line-height:1.45; }
.hold { color:#6b6b68; font-size:.92rem; padding-top:.75rem; }
.goal { border-left:3px solid #000; padding:.15rem 0 .15rem 1rem; margin:0 0 1rem; font-size:.98rem; line-height:1.55; }
.goal .hint { display:block; color:#6b6b68; font-size:.88rem; margin-top:.4rem; }
</style>
"""

st.set_page_config(page_title="Rebalance Planner", page_icon="⚖️", layout="centered")
st.html(CSS)


@st.cache_data
def load_sample_account() -> pd.DataFrame:
    return pd.read_csv(DATA / "sample_account.csv")


@st.cache_data
def load_cov() -> pd.DataFrame:
    return annualized_cov(load_prices(DATA / "prices.csv"))


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
    st.html(f'<div class="step">Step {n} · {label}</div>')


def apply_edits(df: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Replay st.data_editor's pending edits so text above the table can use them."""
    df = df.copy()
    for row, changes in edits.get("edited_rows", {}).items():
        for col, val in changes.items():
            df.iat[int(row), df.columns.get_loc(col)] = val
    df = df.drop(df.index[edits.get("deleted_rows", [])])
    added = pd.DataFrame(edits.get("added_rows", []), columns=df.columns)
    return pd.concat([df, added], ignore_index=True) if len(added) else df


# --- Sidebar: the rules ----------------------------------------------------
st.sidebar.header("Your rules")
st.sidebar.markdown(
    "Prices move every day, so no fund sits exactly on its target. These two dials set "
    "**how much drift you'll tolerate** before a fund gets traded back."
)
abs_band = st.sidebar.slider(
    "Wiggle room for every fund (± percentage points)", 1.0, 15.0, 5.0, 0.5, format="%.1f",
    help="The same allowance for every fund, in percentage points of the account. "
         "Technical name: absolute band.",
) / 100
abs_example = st.sidebar.empty()
rel_band = st.sidebar.slider(
    "Extra check for small funds (± % of the fund's own target)", 5, 100, 25, 5,
    help="Scales the allowance to the fund's size, so a small fund can't double or "
         "vanish while staying inside the first dial. Technical name: relative band.",
) / 100
rel_example = st.sidebar.empty()
st.sidebar.markdown(
    "**The trade-off**\n\n"
    "- **Wider:** fewer trades and lower costs, but the account wanders further from your plan.\n"
    "- **Tighter:** stays closer to plan, but you trade (and pay) more often.\n\n"
    "A fund is traded when it breaks **either** dial."
)
with st.sidebar.expander("How it works"):
    st.markdown(
        "1. **Measure drift.** Compare each fund's share of the account with its target.\n"
        "2. **Flag what's broken.** A fund is off target when it breaks *either* dial. "
        "Whichever dial is stricter for that fund sets its allowed range.\n"
        "3. **Trade only what's broken.** Those funds go back to target. The cash left over goes "
        "to funds that are under target but still in range.\n"
        "4. **Check risk.** Volatility is estimated from a year of daily prices, "
        "before and after the trades."
    )

sample = load_sample_account()
sample = pd.DataFrame({
    "ticker": sample["ticker"],
    "fund": sample["ticker"].map(FUND_NAMES),
    "value": sample["value"],
    "target": sample["target"] * 100,
})
# The editor sits below this text, so replay its edits now to keep the text current.
live = apply_edits(sample, st.session_state.get("account_editor", {}))
live = live.dropna(subset=["ticker"])
live_value = pd.to_numeric(live["value"], errors="coerce").fillna(0).sum()

# --- The goal ------------------------------------------------------------------
def in_sentence(ticker: str) -> str:
    """'International stocks' -> 'international stocks'; leaves 'US bonds' and raw tickers alone."""
    t = str(ticker).strip().upper()
    name = fund(t) or t
    return name if name[:2].isupper() else name[0].lower() + name[1:]


top = live.assign(target=pd.to_numeric(live["target"], errors="coerce")).nlargest(2, "target")
mix = " and ".join(f"{t:g}% in {escape(in_sentence(tk))}" for tk, t in zip(top["ticker"], top["target"]))
st.html(
    '<div class="goal"><b>The goal:</b> keep your account close to the mix you chose, using as few '
    f"trades as possible. Your targets{f' ({mix})' if mix else ''} set how much risk "
    "you're taking. As prices move the mix drifts, and your risk drifts with it. This tool finds "
    "the smallest set of trades that puts it back."
    '<span class="hint">Adjust the dials in the sidebar (tap » on a phone) and watch every '
    "section update.</span></div>"
)

# --- Account (editable) ----------------------------------------------------
with st.expander(f"✏️ Your account · {money(live_value)} across {len(live)} funds (tap to edit)"):
    st.markdown(
        "**Market value** is what each fund is worth today. **Target** is the share of the account "
        "you *want* in it: 40% means 40 cents of every dollar. Targets must add up to 100%."
    )
    account = st.data_editor(
        sample,
        key="account_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        disabled=["fund"],
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "fund": st.column_config.TextColumn("What it is"),
            "value": st.column_config.NumberColumn("Market value", format="dollar", min_value=0),
            "target": st.column_config.NumberColumn("Target", format="%.1f%%", min_value=0, max_value=100),
        },
    )

account = account.dropna(subset=["ticker"]).fillna({"value": 0, "target": 0})
account["ticker"] = account["ticker"].str.strip().str.upper()
account = account[account["ticker"] != ""]
account["target"] = account["target"] / 100

if account["ticker"].duplicated().any():
    st.error("Each ticker can appear only once.")
    st.stop()
if account["value"].sum() <= 0:
    st.error("Enter at least one holding with a market value.")
    st.stop()
target_sum = account["target"].sum()
if abs(target_sum - 1) > 1e-6:
    st.error(f"Your targets add up to {target_sum:.1%}. They must add up to 100%.")
    st.stop()

# --- The math (all in pure modules) ----------------------------------------
holdings = account.set_index("ticker")["value"]
targets = account.set_index("ticker")["target"]
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
abs_example.caption(
    f"Example: **{big}** targets {tb:.0%}, so this dial lets it sit anywhere from "
    f"{max(tb - abs_band, 0):.1%} to {tb + abs_band:.1%}."
)
rel_example.caption(
    f"Example: **{small}** targets {ts:.0%}. The first dial alone would let it run from "
    f"{max(ts - abs_band, 0):.1%} to {ts + abs_band:.1%}. This one holds it to "
    f"{ts * (1 - rel_band):.2%} to {ts * (1 + rel_band):.2%}."
)

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
    risk_error = None
except ValueError as e:
    risk_error = str(e)

# --- Verdict -----------------------------------------------------------------
if n_breach == 0:
    headline = f"All {len(drift)} holdings are inside their ranges. No trades needed."
    sub = "The account is close enough to its target model. Trading now would cost more than it fixes."
else:
    bought = corridor[corridor > 0].sum()
    noun = "holding has" if n_breach == 1 else "holdings have"
    headline = (f"{n_breach} {noun} drifted off target. "
                f"{t_corr['n_trades']} trade{'s' if t_corr['n_trades'] != 1 else ''} fix it.")
    sub = f"Sell {money(bought)} and buy {money(bought)} to bring the account back in line"
    if risk_error is None:
        sub += (f", moving yearly volatility from {vol_before:.2%} to {vol_after:.2%} "
                f"(your target mix is built for {vol_full:.2%})")
    saved = t_full["dollars"] - t_corr["dollars"]
    if saved > 0.5:
        sub += f". That's {money(saved)} less trading than a full rebalance"
    sub += "."

st.html(
    f'<div class="hero"><p class="eyebrow">Rebalance planner · {money(total)} account</p>'
    f"<h1>{escape(headline)}</h1><p>{escape(sub)}</p></div>"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Off target", f"{n_breach} of {len(drift)}", border=True,
          help="Holdings outside their allowed range.")
k2.metric("Trades", t_corr["n_trades"], border=True,
          help="Buys and sells in the proposed plan.")
k3.metric("Money moved", money(t_corr["dollars"]), border=True,
          help="Total of all buys plus all sells.")
if risk_error is None:
    k4.metric("Volatility", f"{vol_after:.1%}", delta=f"{(vol_after - vol_before) * 100:+.2f} pts",
              delta_color="inverse", border=True,
              help="How much the account's value typically swings in a year, after the trades. "
                   f"Your target mix is built for {vol_full:.1%}.")
else:
    k4.metric("Volatility", "n/a", border=True)

# --- Step 1: Where you stand -------------------------------------------------
step(1, "Where you stand")
st.subheader("How far each fund has drifted")
st.markdown(
    "Every fund is lined up on its own target (the **black line**). The **gray bar** is how far "
    "it may drift either way, and the **dot** is where it sits today. "
    "A **red dot** has left its range."
)

order = list(drift.index)
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
for mask, name, color in [(inside, "Inside range", INK), (~inside, "Outside range", BREACH_COLOR)]:
    pts = drift_pts[mask]
    fig.add_scatter(
        name=name, y=[i for i, m in zip(y, mask) if m], x=pts,
        mode="markers+text", text=[f"{v:+.1f} pts" for v in pts],
        textposition="top center",
        textfont=dict(size=11, color=INK),
        marker=dict(size=14, color=color, line=dict(width=2, color="#ffffff")),
        customdata=list(zip(drift.loc[mask, "current_weight"] * 100, drift.loc[mask, "target_weight"] * 100,
                            drift.loc[mask, "breach_reason"].replace("", "inside both bands"))),
        hovertemplate="Now %{customdata[0]:.1f}% vs. target %{customdata[1]:.1f}%<br>%{customdata[2]}"
                      "<extra>" + name + "</extra>",
    )
fig.update_layout(
    height=110 + 62 * len(order), margin=dict(l=0, r=10, t=10, b=0),
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(family="Inter, sans-serif", color=INK),
    legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12)),
    xaxis=dict(title="Distance from target (percentage points)", range=[-reach, reach],
               tickformat="+.0f", gridcolor="#eeeeea", zeroline=False),
    yaxis=dict(tickvals=y, ticktext=labels, range=[len(order) - 0.5, -0.5], showgrid=False),
    hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK)),
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with st.expander("See the numbers"):
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


# --- Step 2: What to trade ----------------------------------------------------
def why(t: str) -> str:
    r = drift.loc[t]
    now, tgt = pct(r["current_weight"]), pct(r["target_weight"])
    pts = abs(r["drift_pts"]) * 100
    side = "over" if r["drift_pts"] > 0 else "under"
    if r["breach"]:
        if r["target_weight"] == 0:
            return f"Not in the target model at all, so it's sold in full."
        if r["breach_reason"] == "absolute":
            return (f"It's {now} of the account against a {tgt} target: {pts:.1f} pts {side}, "
                    f"past the ±{abs_band * 100:g} pt limit. Trading back to target.")
        if r["breach_reason"] == "relative":
            return (f"It's {now} against a {tgt} target. Only {pts:.1f} pts, but that's "
                    f"{abs(r['drift_rel']) * 100:.0f}% {side} its own target, past the ±{rel_band * 100:g}% "
                    f"limit. Small holdings need this second check. Trading back to target.")
        return (f"It's {now} against a {tgt} target: {pts:.1f} pts and "
                f"{abs(r['drift_rel']) * 100:.0f}% {side}, past both limits. Trading back to target.")
    if corridor[t] > 0:
        return (f"Takes a share of the leftover cash. It's {pts:.1f} pts under target but still inside "
                f"its range, so it's topped up rather than forced.")
    return (f"Trimmed to pay for the buys. It's {pts:.1f} pts over target but still inside its range.")


step(2, "What to trade")
if n_breach == 0:
    st.subheader("Nothing to trade")
    st.markdown("Every fund is inside its range. Tighten the bands in the sidebar to see what would trade.")
else:
    st.subheader(f"{t_corr['n_trades']} trades, netting to $0")
    st.markdown(
        "Funds that broke their range go back to target first. Those trades leave cash over "
        "(or short), which is spread across in-range funds that are on the right side of their "
        "target, so nothing is pushed further from where it should be."
    )
    tickets = []
    for t in sorted(order, key=lambda t: -abs(corridor[t])):
        amt = corridor[t]
        if abs(amt) < MIN_TRADE:
            continue
        side = "buy" if amt > 0 else "sell"
        tickets.append(
            f'<div class="ticket"><span class="side {side}">{side.upper()}</span>'
            f'<span class="amt">{money(abs(amt), cents=True)}</span>'
            f'<span class="fund">{escape(t)} <span>{escape(fund(t))}</span></span>'
            f'<span class="why">{escape(why(t))}</span></div>'
        )
    held = [t for t in order if abs(corridor[t]) < MIN_TRADE]
    if held:
        names = ", ".join(escape(t) for t in held)
        them = "them" if len(held) > 1 else "it"
        tickets.append(f'<div class="hold">Left alone: {names}. Still inside '
                       f"{'their ranges' if len(held) > 1 else 'its range'}, so moving {them} "
                       f"wouldn't fix anything that's broken.</div>")
    st.html("".join(tickets))

# --- Step 3: What happens to risk ---------------------------------------------
step(3, "What happens to risk")
if risk_error is not None:
    st.subheader("Risk check unavailable")
    st.warning(f"{risk_error}. The price file covers: {', '.join(cov.index)}.")
else:
    swing_before, swing_after = vol_before * total, vol_after * total
    top = rc_before.idxmax()
    explainer = ("**Volatility** is how much the account's value typically moves in a year. "
                 "In dollars: roughly two years in three, this account should end within "
                 f"**±\u2060{money(swing_before)}** of where it started")
    if t_corr["n_trades"]:
        st.subheader(f"Volatility {vol_before:.1%} → {vol_after:.1%}")
        say(explainer + f" today, and **±\u2060{money(swing_after)}** after the trades.")
    else:
        st.subheader(f"Volatility {vol_before:.1%}, unchanged")
        say(explainer + ". No trades, so no change.")

    direction = "up" if vol_before > vol_full else "down"
    goal = ("The aim isn't the lowest possible risk. It's the risk **your target mix is built for: "
            f"{vol_full:.2%}**. ")
    if t_corr["n_trades"]:
        gap_pts = abs(vol_after - vol_full) * 100
        landing = "right on target" if gap_pts < 0.005 else f"within {gap_pts:.2f} pts of target"
        goal += (f"Drift has pushed the account {direction} to {vol_before:.2%}, and these trades bring it "
                 f"back to {vol_after:.2%}, {landing}.")
    else:
        goal += (f"The account is at {vol_before:.2%} today. Every fund is inside its range, "
                 "so that gap isn't worth paying to close.")
    st.markdown(goal)

    st.markdown(
        f"Risk isn't spread the way money is. **{top}** is {pct(w_before[top])} of the money but "
        f"**{pct(rc_before[top])} of the risk** today"
        + (f", and {pct(rc_after[top])} after the trades." if abs(rc_after[top] - rc_before[top]) > 0.0005 else ".")
    )

    risk_tickers = [t for t in order if t in cov.index]
    rfig = go.Figure()
    for name, rc, color in [("Before", rc_before, BEFORE_COLOR), ("After trades", rc_after, AFTER_COLOR)]:
        rfig.add_bar(
            name=name, y=risk_tickers, x=rc[risk_tickers] * 100, orientation="h",
            marker=dict(color=color, cornerradius=4),
            hovertemplate="%{y}: %{x:.1f}% of risk<extra>" + name + "</extra>",
        )
    rfig.update_layout(
        barmode="group", bargap=0.35, bargroupgap=0.1,
        height=80 + 56 * len(risk_tickers), margin=dict(l=0, r=10, t=30, b=0),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(family="Inter, sans-serif", color=INK),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
        xaxis=dict(title="Share of portfolio risk", ticksuffix="%", gridcolor="#eeeeea", zeroline=False),
        yaxis=dict(autorange="reversed"),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=INK)),
    )
    st.plotly_chart(rfig, width="stretch", config={"displayModeBar": False})

    with st.expander("See the numbers"):
        st.dataframe(
            pd.DataFrame({
                "Fund": [fund(t) for t in risk_tickers],
                "Share of money (before)": w_before[risk_tickers] * 100,
                "Share of risk (before)": rc_before[risk_tickers] * 100,
                "Share of money (after)": w_after[risk_tickers] * 100,
                "Share of risk (after)": rc_after[risk_tickers] * 100,
            }, index=pd.Index(risk_tickers, name="ticker")),
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in [
                "Share of money (before)", "Share of risk (before)",
                "Share of money (after)", "Share of risk (after)"]},
        )
        st.caption("Volatility uses one year of daily adjusted closes (annualized ×252). "
                   "Each fund's share of risk is wᵢ(Σw)ᵢ / σ, and the shares add up to 100%.")

# --- Step 4: Why not rebalance everything? -------------------------------------
step(4, "Why not rebalance everything?")
st.subheader("Corridor vs. full rebalance")
st.markdown(
    "A full rebalance pushes *every* fund back to target, including ones that are only slightly off. "
    "The corridor approach leaves small drifts alone. Each trade has a cost (spreads, commissions, "
    "taxable gains in a taxable account), so skipping trades that barely change anything is worth it."
)
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**Corridor** (this plan)")
    st.metric("Money moved", money(t_corr["dollars"]),
              delta=f"{money(t_corr['dollars'] - t_full['dollars'])} vs. full", delta_color="inverse")
    st.caption(f"{t_corr['n_trades']} trades" + (f" · volatility {vol_after:.2%}" if risk_error is None else ""))
with c2.container(border=True):
    st.markdown("**Full rebalance**")
    st.metric("Money moved", money(t_full["dollars"]))
    st.caption(f"{t_full['n_trades']} trades" + (f" · volatility {vol_full:.2%}" if risk_error is None else ""))

fewer_trades = t_full["n_trades"] - t_corr["n_trades"]
if n_breach == 0:
    say(f"Nothing is out of range, so this plan trades nothing. A full rebalance would still "
                f"move **{money(t_full['dollars'])}** across {t_full['n_trades']} trades.")
elif fewer_trades == 0:
    st.markdown("With bands this tight every fund breaches, so the corridor plan *is* the full "
                "rebalance. Widen the bands to see the difference.")
elif risk_error is None and vol_before - vol_full > 1e-6:
    captured = (vol_before - vol_after) / (vol_before - vol_full)
    say(
        f"The corridor plan gets **{captured:.0%} of the full rebalance's risk reduction** with "
        f"**{fewer_trades} fewer trade{'s' if fewer_trades != 1 else ''}** and "
        f"**{money(t_full['dollars'] - t_corr['dollars'])} less** money moved."
    )
