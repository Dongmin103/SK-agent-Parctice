from __future__ import annotations

from app.ui.presenters import build_consult_view_model, build_executive_view_model


def test_build_consult_view_model_groups_predictions_findings_and_review_status() -> None:
    payload = {
        "selected_agents": ["house", "harvey"],
        "routing_reason": "Mixed safety and regulatory question",
        "predictions": {
            "source": "txgemma",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
            "canonical_smiles": "CCO",
            "signals": [
                {
                    "name": "hERG",
                    "value": "elevated",
                    "unit": None,
                    "confidence": 0.82,
                    "risk_level": "high",
                }
            ],
            "missing_signals": ["bbb"],
            "generated_at": "2026-03-24T12:00:00+00:00",
        },
        "agent_findings": [
            {
                "agent_id": "house",
                "summary": "House sees hERG follow-up risk.",
                "risks": ["hERG"],
                "recommendations": ["Patch clamp"],
                "confidence": 0.81,
                "citations": ["https://example.org/house"],
            }
        ],
        "consulting_answer": "House recommends follow-up.",
        "citations": ["https://example.org/house"],
        "review_required": True,
    }

    view_model = build_consult_view_model(payload)

    assert view_model.selected_agents == ["house", "harvey"]
    assert view_model.review_label == "Human review required"
    assert view_model.prediction_rows[0].label == "hERG"
    assert view_model.prediction_rows[0].value == "elevated"
    assert view_model.missing_signals == ["bbb"]
    assert view_model.findings[0].agent_id == "house"
    assert view_model.citation_count == 1


def test_build_executive_view_model_groups_decision_evidence_and_svg() -> None:
    payload = {
        "canonical_smiles": "N#Cc1ccccc1",
        "molecule_svg": "<svg><rect /></svg>",
        "predictions": {
            "source": "txgemma",
            "target": "KRAS G12C",
            "compound_name": "ABC-101",
            "canonical_smiles": "N#Cc1ccccc1",
            "signals": [
                {
                    "name": "hERG",
                    "value": "elevated",
                    "unit": None,
                    "confidence": 0.82,
                    "risk_level": "high",
                }
            ],
            "missing_signals": [],
            "generated_at": "2026-03-24T12:00:00+00:00",
        },
        "evidence_bundle": {
            "query": "Executive assessment query",
            "packets": {
                "pubmed": {
                    "source": "pubmed",
                    "query": "ABC-101 AND KRAS G12C",
                    "items": [
                        {
                            "source": "pubmed",
                            "pmid": "12345",
                            "title": "ABC-101 evidence",
                            "abstract": "Abstract",
                            "journal": "Journal",
                            "pub_year": 2025,
                            "authors": [],
                            "url": "https://example.org/pubmed",
                            "score": 8.1,
                            "fetched_at": "2026-03-24T12:00:00+00:00",
                            "metadata": {},
                        }
                    ],
                    "fetched_at": "2026-03-24T12:00:00+00:00",
                    "source_health": "ok",
                    "diagnostics": {},
                }
            },
            "items": [],
            "fetched_at": "2026-03-24T12:00:00+00:00",
            "source_health": "ok",
            "missing_sources": [],
            "partial_failures": [],
        },
        "agent_findings": [],
        "executive_summary": "Conditional go until follow-up assays land.",
        "executive_decision": {
            "decision": "conditional_go",
            "rationale": "Safety follow-up is still required.",
            "next_steps": ["Repeat hERG assay"],
        },
        "citations": ["https://example.org/pubmed"],
        "review_required": True,
        "review_reasons": ["Cross-domain review pending"],
    }

    view_model = build_executive_view_model(payload)

    assert view_model.decision == "conditional_go"
    assert view_model.decision_label == "Conditional go"
    assert view_model.molecule_svg == "<svg><rect /></svg>"
    assert view_model.prediction_rows[0].label == "hERG"
    assert view_model.evidence_sources[0].source == "pubmed"
    assert view_model.evidence_sources[0].item_count == 1
    assert view_model.review_reasons == ["Cross-domain review pending"]
