"""Theme tokens, page chrome, nav, and footer for DiamondValue.

Ported from HoopsValue's utils.py theme system and trimmed to this project.
All colors flow from CSS custom-property tokens (var(--panel), var(--fg-1..6),
accents) so every page follows the light/dark flip. inject_theme() emits the
dark base plus a light override when light is active; the chosen theme persists
across full-reload page navigations via the ?theme= URL param.

SENTINEL is defined here ONCE and is the only em dash allowed in string
literals across the codebase (spec rule 1); it renders in table cells for
missing values.
"""
from __future__ import annotations

import streamlit as st

# The single permitted em dash: the missing-value table sentinel.
SENTINEL = "—"

# Ship light by default (config.toml base="light" matches, so iframe components
# render light); dark is opt-in via the nav toggle.
THEME_DEFAULT_DARK = False

# Nav pages: (label, url). Home is rendered separately.
_NAV_PAGES = [
    ("Player", "/Player"),
    ("Accuracy", "/Accuracy"),
    ("About", "/About"),
]

THEME_BASE_CSS = """
<style>
    :root {
        --app-bg:      #0a0f14;
        --bg-base:     #0a0f14;
        --bg-nav:      #080b0e;
        --panel:       rgba(18, 26, 34, 0.60);
        --panel-solid: #12181f;
        --panel-2:     #16202b;
        --panel-hover: rgba(28, 40, 52, 0.9);
        --panel-line:  rgba(90, 110, 125, 0.35);
        --hairline:    rgba(255, 255, 255, 0.08);
        --nav-border:  #1c2630;
        --nav-divider: #2a3742;
        --tint-good:   #12281c;
        --tint-bad:    #2a1618;
        --tint-even:   #16202b;
        --fg-1: #ffffff;
        --fg-2: #cdd6dd;
        --fg-3: #9fb0ba;
        --fg-4: #7f8f99;
        --fg-5: #6c7a83;
        --fg-6: #5a666e;
        /* Ballpark-grass + infield-dirt accents */
        --accent-teal: #17b890;   /* primary accent (outfield green)   */
        --accent-red:  #d1495b;
        --value-good:  #2ecc71;
        --value-bad:   #e2604f;
        --gold:        #e0a63c;    /* base-path clay */
        --blue:        #3d9bd6;
        --amber:       #e0a63c;    /* caveat / model chips */
        --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.38);
        --row-tint:    rgba(255, 255, 255, 0.025);
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
        --app-bg:      linear-gradient(180deg, #fbfcfd 0%, #eef2f0 100%);
        --bg-base:     #f4f7f5;
        --bg-nav:      #ffffff;
        --panel:       #ffffff;
        --panel-solid: #ffffff;
        --panel-2:     #eef2f0;
        --panel-hover: #f0f4f2;
        --panel-line:  #e0e6e2;
        --hairline:    rgba(20, 30, 24, 0.10);
        --nav-border:  #e0e6e2;
        --nav-divider: #c7d0ca;
        --tint-good:   #e9f8ef;
        --tint-bad:    #fceceb;
        --tint-even:   #eef4f0;
        --fg-1: #10241a;
        --fg-2: #33413a;
        --fg-3: #55625b;
        --fg-4: #6e7a73;
        --fg-5: #97a19a;
        --fg-6: #b0b9b3;
        --accent-teal: #0e9e79;
        --value-good:  #16a34a;
        --value-bad:   #d1402f;
        --gold:        #9a6a00;
        --amber:       #a8730a;
        --blue:        #2471a3;
        --shadow-card: 0 1px 2px rgba(20,30,24,.06), 0 4px 14px rgba(20,30,24,.07);
        --row-tint:    rgba(20, 30, 24, 0.028);
    }
</style>
"""

COMMON_CSS = """
<style>
    .block-container { padding-top: 2.2rem; max-width: 1180px; }
    #MainMenu, header[data-testid="stHeader"] { visibility: hidden; }
    /* We use the custom top nav; hide Streamlit's default multipage sidebar. */
    [data-testid="stSidebarNav"], [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    .top-nav {
        display: flex; align-items: center; gap: 0.15rem; flex-wrap: wrap;
        padding: 0.2rem 0 0.9rem; margin-bottom: 0.8rem;
        border-bottom: 1px solid var(--nav-border); font-size: 0.94rem;
    }
    .top-nav a {
        color: var(--fg-3); text-decoration: none; padding: 0.2rem 0.6rem;
        border-radius: 6px; font-weight: 500;
    }
    .top-nav a:hover { color: var(--fg-1); background: var(--panel-hover); }
    .top-nav a.active { color: var(--accent-teal); font-weight: 700; }
    .top-nav a.home-link { color: var(--fg-2); font-weight: 700; }
    .top-nav .divider { color: var(--nav-divider); margin: 0 0.3rem; }
    .dv-brand {
        font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em;
        color: var(--fg-1); margin: 0 0 0.1rem;
    }
    .dv-brand .accent { color: var(--accent-teal); }
    .dv-tagline { color: var(--fg-4); font-size: 0.95rem; margin-bottom: 0.6rem; }
    .dv-badge {
        display: inline-block; font-size: 0.72rem; font-weight: 700;
        padding: 0.1rem 0.5rem; border-radius: 999px; letter-spacing: 0.02em;
        text-transform: uppercase; vertical-align: middle;
    }
    .dv-badge.confirmed { background: var(--tint-good); color: var(--value-good); }
    .dv-badge.projected { background: var(--tint-even); color: var(--amber); }
    .dv-note { color: var(--fg-4); font-size: 0.85rem; }
    .dv-footer {
        margin-top: 3.5rem; padding-top: 1.3rem; font-size: 0.82rem;
        border-top: 1px solid var(--panel-line); color: var(--fg-5);
        text-align: center; line-height: 1.5;
    }
    .dv-footer a { color: var(--fg-3); }
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
    st.button("Light mode" if dark else "Dark mode", key="theme_toggle_btn",
              on_click=_flip, help="Toggle light/dark")
    return st.session_state.get("theme_dark", THEME_DEFAULT_DARK)


def render_page_chrome() -> None:
    """One-call chrome: theme tokens + shared CSS. Call once after
    st.set_page_config on every page.
    """
    inject_theme()
    st.markdown(COMMON_CSS, unsafe_allow_html=True)


def render_nav(current: str) -> None:
    """Top nav bar. `current` matches a label in _NAV_PAGES (or "Home").

    Links use target="_self" (the default), NOT "_top": Streamlit Community
    Cloud serves the app inside an iframe, and "_top" would navigate the outer
    wrapper (wrong origin) instead of the app frame, so the links appear dead.
    "_self" navigates the app's own frame, which works whether the app is
    iframed (Streamlit Cloud) or served directly (local/Render).
    """
    links = '<a class="home-link" href="/" target="_self">Home</a>'
    links += '<span class="divider">|</span>'
    for label, url in _NAV_PAGES:
        cls = "active" if label == current else ""
        links += f'<a class="{cls}" href="{url}" target="_self">{label}</a>'
    st.markdown(f'<div class="top-nav">{links}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Site-wide footer with the required MLBAM attribution. Call at the bottom
    of every page. Year is stamped at render time.
    """
    import datetime as _dt

    year = _dt.date.today().year
    html = (
        '<div class="dv-footer">'
        "Statistics data via MLB Stats API. "
        f"Copyright {year} MLB Advanced Media, L.P. Use of any content "
        "acknowledges agreement to the terms at "
        '<a href="http://gdx.mlb.com/components/copyright.txt" target="_blank" '
        'rel="noopener">gdx.mlb.com/components/copyright.txt</a>.<br>'
        "DiamondValue is an independent, non-commercial project and is not "
        "affiliated with or endorsed by Major League Baseball."
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def theme_fig(fig):
    """Make a Plotly figure follow the active theme (charts can't read CSS
    vars). Call inline at the plot site.
    """
    dark = st.session_state.get("theme_dark", THEME_DEFAULT_DARK)
    axis = "#cdd6dd" if dark else "#33413a"
    grid = "rgba(255,255,255,0.08)" if dark else "rgba(20,30,24,0.10)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=axis),
        hoverlabel=dict(
            bgcolor="#12181f" if dark else "#ffffff",
            font=dict(color="#e8eef0" if dark else "#10241a"),
        ),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return fig
