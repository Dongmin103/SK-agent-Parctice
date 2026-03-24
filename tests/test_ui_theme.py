from __future__ import annotations

from app.ui.theme import build_theme_css


def test_build_theme_css_includes_explicit_contrast_rules_for_core_widgets() -> None:
    css = build_theme_css()

    assert '[data-testid="stMetricValue"]' in css
    assert '[data-testid="stMetricLabel"]' in css
    assert '[data-baseweb="tab"][aria-selected="true"]' in css
    assert '.stTextInput label p' in css
    assert '.stTextArea label p' in css
    assert '[data-baseweb="input"]' in css
    assert 'color: var(--olive) !important;' in css
    assert 'color: var(--paper) !important;' in css


def test_build_theme_css_includes_sidebar_specific_contrast_overrides() -> None:
    css = build_theme_css()

    assert '[data-testid="stSidebar"] .stTextInput label p' in css
    assert '[data-testid="stSidebar"] .stRadio label p' in css
    assert '[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]' in css
    assert '[data-testid="stSidebar"] [data-testid="stCaptionContainer"]' in css
    assert 'opacity: 1 !important;' in css
