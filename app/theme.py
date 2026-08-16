"""
Shared visual theme for the Causal Uplift Marketing Intelligence console.

Design concept: "Trial Console." The whole project is organized around one
idea — treatment vs. control — so that idea becomes the organizing visual
element too. Every page opens with a live split-bar built from that page's
own real numbers, dividing "signal" (pine green) from "baseline" (stone
grey). Each page spends exactly one gold accent on the single number that
matters most there.

Drop this file at the project root (next to streamlit_app.py's ROOT dir,
i.e. wherever `artifacts/` lives) and import it from every page with:

    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from theme import inject_theme, eyebrow, rule, split_bar, plotly_layout
"""

import textwrap

import streamlit as st

# ---------------------------------------------------------------- palette --
INK = "#1A2420"     # primary text
PAPER = "#F1F3EE"   # page background
CARD = "#FFFFFF"    # card surface
LINE = "#D9DCD3"    # hairline borders
PINE = "#2F6B52"    # treatment / signal
STONE = "#8A8477"   # control / baseline
GOLD = "#AD8A2E"    # single highlight accent — spend it on one thing per page
RUST = "#B5482F"    # negative / caution, used sparingly

FONT_HREF = (
    "https://fonts.googleapis.com/css2?"
    "family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;"
    "1,6..72,400&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
)


def _no_blank_lines(html: str) -> str:
    """Markdown treats a blank line as the end of a raw HTML block, so any
    blank line inside our <style> block causes everything after it to be
    rendered as literal text instead of CSS. Strip blank lines to keep the
    whole block recognized as one continuous HTML block."""
    return "\n".join(line for line in html.splitlines() if line.strip() != "")


def inject_theme():
    """Call once near the top of every page, right after st.set_page_config (if present)."""
    st.markdown(
        _no_blank_lines(textwrap.dedent(f"""\
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{FONT_HREF}" rel="stylesheet">
        <style>
        :root {{
            --ink: {INK}; --paper: {PAPER}; --card: {CARD}; --line: {LINE};
            --pine: {PINE}; --stone: {STONE}; --gold: {GOLD}; --rust: {RUST};
        }}

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: var(--ink); }}
        .stApp {{ background: var(--paper); }}

        h1 {{
            font-family: 'Newsreader', serif !important;
            font-weight: 500 !important;
            font-size: 2.4rem !important;
            letter-spacing: -0.01em;
            color: var(--ink) !important;
            margin-bottom: 0.2rem !important;
        }}
        h2, h3 {{
            font-family: 'Newsreader', serif !important;
            font-weight: 500 !important;
            color: var(--ink) !important;
        }}

        .console-eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--stone);
            margin-bottom: 0.35rem;
        }}
        .console-eyebrow b {{ color: var(--pine); }}

        .console-rule {{ border: none; border-top: 1px solid var(--line); margin: 1.4rem 0; }}

        [data-testid="stMetric"] {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 2px;
            padding: 0.9rem 1rem 0.7rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.68rem !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--stone) !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: var(--ink) !important;
            font-weight: 600 !important;
        }}

        [data-testid="stDataFrame"] {{ border: 1px solid var(--line); }}

        [data-testid="stSidebar"] {{ background: var(--ink); }}
        [data-testid="stSidebar"] * {{ color: var(--paper) !important; }}
        [data-testid="stSidebar"] .console-eyebrow {{ color: #9AA69C; }}

        .stButton>button {{
            font-family: 'IBM Plex Mono', monospace;
            letter-spacing: 0.04em;
            background: var(--ink);
            color: var(--paper);
            border: 1px solid var(--ink);
            border-radius: 2px;
        }}
        .stButton>button:hover {{
            background: var(--pine); border-color: var(--pine); color: white;
        }}

        [data-testid="stAlert"] {{
            background: var(--card);
            border: 1px solid var(--line);
            border-left: 3px solid var(--gold);
            border-radius: 2px;
        }}

        .split-bar-wrap {{ margin: 0.4rem 0 1.6rem 0; }}
        .split-bar {{
            display: flex; width: 100%; height: 10px; border-radius: 2px;
            overflow: hidden; border: 1px solid var(--line);
        }}
        .split-bar-caption {{
            display: flex; justify-content: space-between;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.7rem; color: var(--stone); margin-top: 0.35rem;
        }}
        </style>
        """)),
        unsafe_allow_html=True,
    )


def eyebrow(text: str):
    """Small mono-caps label placed above a page title, e.g. 'ARM COMPARISON · MEN'S VS WOMEN'S'."""
    st.markdown(f'<div class="console-eyebrow">{text}</div>', unsafe_allow_html=True)


def rule():
    st.markdown('<hr class="console-rule">', unsafe_allow_html=True)


def split_bar(left_label, left_value, right_label, right_value,
              left_color=PINE, right_color=STONE, unit="", decimals=1):
    """
    The console's signature element: a thin bar split at the true
    proportion between two real numbers pulled from the page's own data
    (treatment vs. control customers, targeted vs. untargeted, etc.).
    """
    total = max(left_value + right_value, 1e-9)
    left_pct = 100 * left_value / total
    right_pct = 100 - left_pct
    st.markdown(
        textwrap.dedent(f"""\
        <div class="split-bar-wrap">
          <div class="split-bar">
            <div style="width:{left_pct:.2f}%; background:{left_color};"></div>
            <div style="width:{right_pct:.2f}%; background:{right_color};"></div>
          </div>
          <div class="split-bar-caption">
            <span>{left_label} · {left_value:,.{decimals}f}{unit} ({left_pct:.0f}%)</span>
            <span>{right_label} · {right_value:,.{decimals}f}{unit} ({right_pct:.0f}%)</span>
          </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height=420):
    """Apply the console's chart styling to any plotly figure. Returns the figure."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        title_font=dict(family="Newsreader, serif", size=18, color=INK),
        plot_bgcolor=CARD,
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=[PINE, STONE, GOLD, RUST, "#5B7A8C", "#C9C3B2"],
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig