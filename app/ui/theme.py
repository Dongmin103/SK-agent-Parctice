from __future__ import annotations


def build_theme_css() -> str:
    return """
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
      --ink: #16211c;
      --olive: #20372c;
      --olive-soft: #42584d;
      --sand: #f7f4eb;
      --paper: #fffdf7;
      --paper-strong: #fffaf2;
      --accent: #d85d2a;
      --accent-soft: #f4b183;
      --mint: #86b3a2;
      --line: rgba(22, 33, 28, 0.12);
      --shadow: 0 18px 46px rgba(52, 46, 27, 0.10);
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
      background:
        radial-gradient(circle at top right, rgba(216, 93, 42, 0.16), transparent 28%),
        radial-gradient(circle at top left, rgba(134, 179, 162, 0.18), transparent 24%),
        linear-gradient(180deg, #f5f0e5 0%, #fbf8f0 48%, #f7f2e8 100%);
      color: var(--ink);
      font-family: "IBM Plex Sans KR", sans-serif;
    }

    [data-testid="stMainBlockContainer"] {
      padding-top: 2.4rem;
      padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, rgba(22, 33, 28, 0.95), rgba(32, 55, 44, 0.97));
    }

    [data-testid="stSidebar"] * {
      color: #f8f4e8 !important;
    }

    .hero {
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255, 253, 247, 0.94), rgba(255, 245, 226, 0.84));
      border-radius: 28px;
      padding: 28px 30px 24px 30px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }

    .eyebrow, h1, h2, h3, [data-testid="stMetricLabel"], [data-baseweb="tab"] p {
      font-family: "Space Grotesk", sans-serif;
      letter-spacing: -0.02em;
    }

    h1, h2, h3, p, li, label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
      color: var(--ink);
    }

    .eyebrow {
      color: var(--accent) !important;
      text-transform: uppercase;
      font-size: 0.8rem;
      margin-bottom: 0.3rem;
    }

    .hero h1 {
      margin: 0;
      color: var(--olive) !important;
      font-size: 2.6rem;
    }

    .hero-copy {
      max-width: 62rem;
      margin-top: 0.8rem;
      color: rgba(22, 33, 28, 0.86) !important;
      line-height: 1.6;
    }

    [data-testid="stForm"] {
      background: rgba(255, 253, 247, 0.74);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 1rem 1rem 1.2rem 1rem;
      box-shadow: 0 10px 24px rgba(52, 46, 27, 0.06);
      margin-bottom: 1.2rem;
    }

    .stTextInput label p,
    .stTextArea label p,
    .stRadio label p,
    .stSelectbox label p {
      color: var(--olive) !important;
      font-weight: 600;
    }

    [data-testid="stSidebar"] .stTextInput label p,
    [data-testid="stSidebar"] .stTextArea label p,
    [data-testid="stSidebar"] .stRadio label p,
    [data-testid="stSidebar"] .stSelectbox label p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] [role="radiogroup"] label *,
    [data-testid="stSidebar"] [data-baseweb="radio"] label,
    [data-testid="stSidebar"] [data-baseweb="radio"] label * {
      color: var(--paper) !important;
      opacity: 1 !important;
    }

    [data-baseweb="input"],
    [data-baseweb="base-input"],
    textarea {
      background: rgba(255, 250, 242, 0.96) !important;
      color: var(--ink) !important;
      border-color: rgba(32, 55, 44, 0.18) !important;
    }

    [data-baseweb="input"] input,
    [data-baseweb="base-input"] input,
    textarea {
      color: var(--ink) !important;
      -webkit-text-fill-color: var(--ink) !important;
      caret-color: var(--accent) !important;
    }

    [data-baseweb="input"]:focus-within,
    [data-baseweb="base-input"]:focus-within,
    textarea:focus {
      border-color: rgba(216, 93, 42, 0.45) !important;
      box-shadow: 0 0 0 1px rgba(216, 93, 42, 0.18) !important;
    }

    [data-testid="stMetric"] {
      background: rgba(255, 253, 247, 0.90);
      border: 1px solid var(--line);
      padding: 14px 16px;
      border-radius: 18px;
      box-shadow: 0 10px 26px rgba(52, 46, 27, 0.08);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
      color: var(--olive) !important;
      opacity: 1 !important;
    }

    [data-testid="stMetricValue"] {
      font-size: 2.1rem;
      font-weight: 700;
    }

    .callout-card {
      margin-top: 8px;
      margin-bottom: 18px;
      border-left: 6px solid var(--accent);
      background: rgba(255, 250, 240, 0.92);
      border-radius: 18px;
      padding: 16px 18px;
    }

    .callout-label {
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      font-size: 0.9rem;
      color: var(--accent) !important;
    }

    .callout-copy {
      margin: 0.35rem 0 0 0;
      color: var(--ink) !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
      background: linear-gradient(135deg, #d85d2a 0%, #ef8b54 100%);
      color: var(--paper) !important;
      border: none;
      border-radius: 999px;
      font-family: "Space Grotesk", sans-serif;
      font-weight: 700;
      min-height: 3rem;
      box-shadow: 0 12px 28px rgba(216, 93, 42, 0.24);
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
      filter: saturate(1.02) brightness(0.98);
    }

    .stTabs [data-baseweb="tab-list"] {
      gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
      background: rgba(255, 253, 247, 0.88);
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 0.45rem 1rem;
      color: var(--olive) !important;
    }

    .stTabs [data-baseweb="tab"] *,
    .stTabs [data-baseweb="tab"] p {
      color: var(--olive) !important;
      opacity: 1 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
      background: linear-gradient(135deg, rgba(32, 55, 44, 0.96), rgba(46, 77, 61, 0.96));
      border-color: rgba(32, 55, 44, 0.96);
      box-shadow: 0 8px 18px rgba(32, 55, 44, 0.16);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] *,
    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
      color: var(--paper) !important;
    }

    [data-testid="stExpander"] {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 253, 247, 0.78);
    }

    [data-testid="stDataFrame"] {
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }

    .stAlert {
      border-radius: 18px;
    }
    """
