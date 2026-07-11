"""Theme tokens, page chrome, nav, and footer for DiamondValue.

The visual system is HoopsValue's, ported wholesale (Barrett's NBA site at
nba-value-app): the same surface/text/accent tokens in dark and light, the
same Space Grotesk + Manrope self-hosted webfonts, the fixed pill nav with the
red active state, and the pinned theme toggle. Only the wordmark text and the
handful of baseball-specific components (game cards, badges) are new.

SENTINEL is defined here ONCE and is the only em dash allowed in string
literals across the codebase (spec rule 1); it renders in table cells for
missing values.
"""
from __future__ import annotations

import streamlit as st

# The single permitted em dash: the missing-value table sentinel.
SENTINEL = "—"

# Ship light by default (config.toml base="light" matches, so iframe components
# render light); dark is opt-in via the pinned toggle, persisted via ?theme=.
THEME_DEFAULT_DARK = False

# Nav pages: (label, url). Home is rendered separately.
_NAV_PAGES = [
    ("Player", "/Player"),
    ("Accuracy", "/Accuracy"),
    ("About", "/About"),
]

# ── Tokens: HoopsValue's exact palette ───────────────────────────────────────
THEME_BASE_CSS = """
<style>
    :root {
        /* surfaces */
        --app-bg:      #0a0a14;
        --bg-base:     #0a0a14;
        --bg-nav:      #0a0a0a;
        --panel:       rgba(20, 20, 42, 0.55);
        --panel-solid: #15171d;
        --panel-2:     #1a1a2e;
        --panel-hover: rgba(30, 30, 56, 0.85);
        --panel-line:  rgba(80, 80, 110, 0.35);
        --hairline:    rgba(255, 255, 255, 0.08);
        --hairline-soft: rgba(255, 255, 255, 0.04);
        --nav-border:  #222;
        --nav-divider: #333;
        /* tinted value-card surfaces */
        --tint-good:   #1a2e1a;
        --tint-bad:    #2e1a1a;
        --tint-even:   #1a1a2e;
        /* text ramp */
        --fg-1: #ffffff;
        --fg-2: #cdcdd5;
        --fg-3: #aaaaaa;
        --fg-4: #8a8a93;
        --fg-5: #777777;
        --fg-6: #666666;
        /* brand accents */
        --accent-red:  #e63946;
        --accent-teal: #16d4c1;
        --value-good:  #2ecc71;
        --value-bad:   #e74c3c;
        --gold:        #f1c40f;
        --blue:        #3498db;
        --orange:      #f39c12;
        --purple:      #9b59b6;
        --sky:         #7ec8e8;
        --amber:       #f0b35b;
        /* elevation + table polish */
        --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.35);
        --row-tint: rgba(255, 255, 255, 0.025);
        --bar-tint: rgba(22, 212, 193, 0.16);
    }
    html, body, .stApp { background: var(--app-bg) !important; }
    .stApp, body { color: var(--fg-2); }
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 { color: var(--fg-1) !important; }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label,
    [data-baseweb="form-control-label"] { color: var(--fg-2) !important; }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-1) !important;
    }
    div[data-baseweb="select"] svg { fill: var(--fg-3) !important; }
    ul[data-baseweb="menu"], div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] [role="listbox"] { background: var(--panel-solid) !important; }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] *,
    ul[data-baseweb="menu"] li { color: var(--fg-2) !important; }
    [data-baseweb="popover"] [role="option"]:hover,
    ul[data-baseweb="menu"] li:hover { background: var(--panel-hover) !important; }
    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-baseweb="input"] {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
    }
    [data-testid="stDateInput"] input,
    [data-testid="stTextInput"] input,
    div[data-baseweb="input"] input { color: var(--fg-1) !important; }
    [data-testid="stExpander"] details {
        background: var(--panel) !important;
        border-color: var(--panel-line) !important;
    }
    [data-testid="stButton"] button, .stButton button {
        background: var(--panel-solid) !important;
        border-color: var(--panel-line) !important;
        color: var(--fg-2) !important;
    }
    [data-testid="stButton"] button:hover, .stButton button:hover {
        background: var(--panel-hover) !important;
        border-color: var(--fg-5) !important;
        color: var(--fg-1) !important;
    }
</style>
"""

THEME_LIGHT_CSS = """
<style>
    :root {
        --app-bg:      linear-gradient(180deg, #fbfcfd 0%, #eef1f4 100%);
        --bg-base:     #f4f6f8;
        --bg-nav:      #ffffff;
        --panel:       #ffffff;
        --panel-solid: #ffffff;
        --panel-2:     #eef1f4;
        --panel-hover: #f1f3f6;
        --panel-line:  #e3e6eb;
        --hairline:    rgba(20, 22, 40, 0.10);
        --hairline-soft: rgba(20, 22, 40, 0.05);
        --nav-border:  #e3e6eb;
        --nav-divider: #c9ccd3;
        --tint-good:   #eafaf1;
        --tint-bad:    #fdeceb;
        --tint-even:   #eef3f8;
        --fg-1: #14142a;
        --fg-2: #3a3d48;
        --fg-3: #585c68;
        --fg-4: #71757f;
        --fg-5: #9aa0ab;
        --fg-6: #b3b8c2;
        --accent-teal: #0fae9d;
        --value-good:  #16a34a;
        --value-bad:   #dc3a2c;
        --gold:        #9a6a00;
        --amber:       #a8730a;
        --orange:      #b45f06;
        --purple:      #7d3fa8;
        --blue:        #2471a3;
        --sky:         #146c94;
        --shadow-card: 0 1px 2px rgba(20,22,40,.06), 0 4px 14px rgba(20,22,40,.07);
        --row-tint: rgba(20, 22, 40, 0.028);
        --bar-tint: rgba(15, 174, 157, 0.15);
    }
</style>
"""

# ── Fonts + shared chrome (HoopsValue's Space Grotesk / Manrope system) ──────
COMMON_CSS = """
<style>
    @font-face{font-family:'Space Grotesk';font-style:normal;font-weight:500;font-display:swap;
      src:url('/app/static/fonts/space-grotesk-500.woff2') format('woff2');}
    @font-face{font-family:'Space Grotesk';font-style:normal;font-weight:600;font-display:swap;
      src:url('/app/static/fonts/space-grotesk-600.woff2') format('woff2');}
    @font-face{font-family:'Space Grotesk';font-style:normal;font-weight:700;font-display:swap;
      src:url('/app/static/fonts/space-grotesk-700.woff2') format('woff2');}
    @font-face{font-family:'Manrope';font-style:normal;font-weight:500;font-display:swap;
      src:url('/app/static/fonts/manrope-500.woff2') format('woff2');}
    @font-face{font-family:'Manrope';font-style:normal;font-weight:600;font-display:swap;
      src:url('/app/static/fonts/manrope-600.woff2') format('woff2');}
    @font-face{font-family:'Manrope';font-style:normal;font-weight:700;font-display:swap;
      src:url('/app/static/fonts/manrope-700.woff2') format('woff2');}

    html, body, .stApp, [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"], button, input {
        font-family: 'Manrope', -apple-system, sans-serif;
    }
    [data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3, .dv-brand, .dv-game-match {
        font-family: 'Space Grotesk', 'Manrope', sans-serif;
    }

    /* Clear the fixed nav bar */
    .block-container { padding-top: 4.4rem; max-width: 1180px; }
    #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
    [data-testid="stToolbar"]        { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    [data-testid="stStatusWidget"]   { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stSidebarNav"], [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* Fixed top nav bar: HoopsValue's pill nav, red active state */
    .top-nav {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0 1.5rem;
        padding-right: 3.5rem;
        height: 3rem;
        background: var(--bg-nav);
        border-bottom: 1px solid var(--nav-border);
        flex-wrap: nowrap;
    }
    .top-nav a {
        text-decoration: none;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--fg-3);
        border: 1px solid transparent;
        transition: all 0.15s;
        white-space: nowrap;
    }
    .top-nav a:hover { border-color: var(--accent-red); color: var(--fg-1); text-decoration: none; }
    .top-nav a.active { background: var(--accent-red); border-color: var(--accent-red); color: #fff; }
    .top-nav .home-link {
        color: var(--fg-6);
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.3rem 0.7rem;
        margin-right: 0.25rem;
        border: none;
    }
    .top-nav .home-link:hover { color: var(--fg-1); border: none; }
    .top-nav .divider { color: var(--nav-divider); font-size: 0.75rem; margin: 0 0.1rem; user-select: none; }
    @media (max-width: 760px) {
        .top-nav { overflow-x: auto; overflow-y: hidden; scrollbar-width: none; padding-right: 4rem; }
        .top-nav::-webkit-scrollbar { display: none; }
        .top-nav::after { content: ""; flex: 0 0 8rem; }
    }

    /* Pinned theme toggle (top-right, inside the nav bar row) */
    .st-key-dv_theme_toggle {
        position: fixed;
        top: 0.35rem; right: 0.9rem;
        z-index: 10000;
        width: auto !important;
    }
    .st-key-dv_theme_toggle button {
        padding: 0.15rem 0.7rem !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        border-radius: 20px !important;
    }

    /* Brand + hero */
    .dv-brand {
        font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em;
        color: var(--fg-1); margin: 0 0 0.1rem;
    }
    .dv-brand .accent { color: var(--accent-teal); }
    .dv-tagline {
        color: var(--fg-4); font-size: 0.82rem; margin-bottom: 0.6rem;
        font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
    }
    .dv-badge {
        display: inline-block; font-size: 0.72rem; font-weight: 700;
        padding: 0.1rem 0.5rem; border-radius: 999px; letter-spacing: 0.02em;
        text-transform: uppercase; vertical-align: middle;
    }
    .dv-badge.confirmed { background: var(--tint-good); color: var(--value-good); }
    .dv-badge.projected { background: var(--tint-even); color: var(--amber); }
    .dv-note { color: var(--fg-4); font-size: 0.85rem; }

    /* Clickable game cards */
    a.dv-game-card {
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; flex-wrap: wrap;
        padding: 0.85rem 1.15rem; margin-bottom: 0.6rem;
        background: var(--panel); border: 1px solid var(--panel-line);
        border-radius: 12px; text-decoration: none;
        box-shadow: var(--shadow-card); transition: border-color .12s, transform .12s;
    }
    a.dv-game-card:hover { border-color: var(--accent-teal); transform: translateY(-1px); }
    .dv-game-match { font-size: 1.05rem; font-weight: 700; color: var(--fg-1); }
    .dv-game-match .at { color: var(--accent-teal); font-weight: 500; margin: 0 0.3rem; }
    .dv-game-right { display: flex; align-items: center; gap: 0.9rem; }
    .dv-game-time { color: var(--fg-3); font-size: 0.9rem; }
    .dv-game-arrow { color: var(--accent-teal); font-weight: 700; }
    a.dv-back {
        display: inline-block; color: var(--accent-teal); text-decoration: none;
        font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;
    }
    a.dv-back:hover { filter: brightness(1.1); }

    /* Footer: HoopsValue's layout adapted */
    .dv-footer { margin-top: 3.5rem; padding-top: 1.4rem; font-size: 0.85rem; }
    .dv-foot-disc { text-align: center; color: var(--fg-5); font-size: 0.78rem;
        line-height: 1.45; max-width: 760px; margin: 0 auto 1.2rem; }
    .dv-foot-rule { border-top: 1px solid var(--panel-line); margin: 0 0 1rem; }
    .dv-foot-bottom { display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 0.5rem 1.1rem; color: var(--fg-5); }
    .dv-foot-bottom a { color: var(--fg-3); text-decoration: none; }
    .dv-foot-bottom a:hover { color: var(--fg-1); }
</style>
"""


def inject_theme() -> None:
    """Emit theme tokens (dark base + light override when active). Call once per
    page before any token-referencing CSS. Theme persists across navigations via
    ?theme=.
    """
    if "theme_dark" not in st.session_state:
        qp = st.query_params.get("theme")
        st.session_state["theme_dark"] = (
            (qp == "dark") if qp in ("dark", "light") else THEME_DEFAULT_DARK
        )
    st.markdown(THEME_BASE_CSS, unsafe_allow_html=True)
    if not st.session_state.get("theme_dark", THEME_DEFAULT_DARK):
        st.markdown(THEME_LIGHT_CSS, unsafe_allow_html=True)


def render_theme_toggle() -> bool:
    """Light/dark toggle backed by st.session_state['theme_dark']. Mirrors the
    choice into ?theme= so a full-reload navigation carries it. Returns dark.
    """
    st.session_state.setdefault("theme_dark", THEME_DEFAULT_DARK)

    def _flip():
        new_dark = not st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
        st.session_state["theme_dark"] = new_dark
        st.query_params["theme"] = "dark" if new_dark else "light"

    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    st.button("Light" if dark else "Dark", key="theme_toggle_btn",
              on_click=_flip, help="Toggle light/dark")
    return st.session_state.get("theme_dark", THEME_DEFAULT_DARK)


def render_page_chrome() -> None:
    """One-call chrome: theme tokens + shared CSS. Call once after
    st.set_page_config on every page.
    """
    inject_theme()
    st.markdown(COMMON_CSS, unsafe_allow_html=True)


def render_nav(current: str) -> None:
    """Fixed top nav bar plus the pinned theme toggle. `current` matches a
    label in _NAV_PAGES (or "Home").

    Links use target="_self" (the default), NOT "_top": Streamlit Community
    Cloud serves the app inside an iframe, and "_top" would navigate the outer
    wrapper (wrong origin) instead of the app frame, so the links appear dead.
    "_self" navigates the app's own frame, which works whether the app is
    iframed (Streamlit Cloud) or served directly (local/Render).
    """
    home_cls = "active" if current == "Home" else ""
    links = f'<a class="home-link {home_cls}" href="/" target="_self">DiamondValue</a>'
    links += '<span class="divider">|</span>'
    for label, url in _NAV_PAGES:
        cls = "active" if label == current else ""
        links += f'<a class="{cls}" href="{url}" target="_self">{label}</a>'
    st.markdown(f'<div class="top-nav">{links}</div>', unsafe_allow_html=True)
    # Pinned top-right toggle, on every page (the HoopsValue pattern: a keyed
    # container that COMMON_CSS position:fixes into the nav row).
    with st.container(key="dv_theme_toggle"):
        render_theme_toggle()


def render_footer() -> None:
    """Site-wide footer with the required MLBAM attribution. Call at the bottom
    of every page. Year is stamped at render time.
    """
    import datetime as _dt

    year = _dt.date.today().year
    html = (
        '<div class="dv-footer">'
        '<div class="dv-foot-disc">Statistics data via MLB Stats API. '
        f"Copyright {year} MLB Advanced Media, L.P. Use of any content "
        "acknowledges agreement to the terms at "
        '<a href="http://gdx.mlb.com/components/copyright.txt" target="_blank" '
        'rel="noopener">gdx.mlb.com/components/copyright.txt</a>.</div>'
        '<div class="dv-foot-rule"></div>'
        '<div class="dv-foot-bottom">'
        f"<div>&copy; {year} DiamondValue. Every number is an expected value.</div>"
        '<div><a href="/About" target="_self">About</a></div>'
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def theme_fig(fig):
    """Make a Plotly figure follow the active theme (charts can't read CSS
    vars). Call inline at the plot site.
    """
    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    axis = "#cdcdd5" if dark else "#3a3d48"
    grid = "rgba(255,255,255,0.08)" if dark else "rgba(20,22,40,0.10)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=axis, family="Manrope, sans-serif"),
        hoverlabel=dict(
            bgcolor="#1a1a2e" if dark else "#ffffff",
            font=dict(color="#e8e8f0" if dark else "#14142a"),
        ),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig
