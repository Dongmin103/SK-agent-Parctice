from __future__ import annotations

from app.agents.ceo_synthesizer import CEOSynthesizer
from app.agents.harvey_agent import HarveyAgent, build_bedrock_harvey_agent
from app.agents.house_agent import HouseAgent, build_bedrock_house_agent
from app.agents.pubmed_query_agent import build_bedrock_pubmed_query_agent
from app.agents.router_agent import ConsultRouterAgent
from app.agents.walter_agent import WalterAgent, build_bedrock_walter_agent
from app.api.settings import AppSettings
from app.clients.chembl import ChEMBLClient
from app.clients.clinicaltrials import ClinicalTrialsClient
from app.clients.evidence_coordinator import EvidenceCoordinator
from app.clients.openfda import OpenFDAClient
from app.clients.pubchem import PubChemClient
from app.clients.pubmed import PubMedClient
from app.clients.txgemma import TxGemmaClient, build_txgemma_client
from app.domain.models import PredictionBundle
from app.workflows.consult import ConsultWorkflow
from app.workflows.executive import ExecutiveWorkflow


class StubPredictionClient:
    """Allow local API boot and smoke tests without a live TxGemma endpoint."""

    def predict(
        self,
        *,
        smiles: str,
        target: str | None = None,
        compound_name: str | None = None,
    ) -> PredictionBundle:
        return PredictionBundle(
            source="txgemma_stub",
            target=target,
            compound_name=compound_name,
            canonical_smiles=smiles,
            signals=[],
            missing_signals=[],
        )


def build_consult_workflow(settings: AppSettings) -> ConsultWorkflow:
    evidence_coordinator = _build_evidence_coordinator(settings)
    walter_agent = _build_walter_agent(settings)
    house_agent = _build_house_agent(settings)
    harvey_agent = _build_harvey_agent(settings)
    router_agent = _build_router_agent(settings)

    return ConsultWorkflow(
        prediction_client=_build_prediction_client(settings),
        router_agent=router_agent,
        evidence_coordinator=evidence_coordinator,
        walter_agent=walter_agent,
        house_agent=house_agent,
        harvey_agent=harvey_agent,
        retmax=settings.retmax,
        top_k=settings.top_k,
    )


def build_executive_workflow(settings: AppSettings) -> ExecutiveWorkflow:
    evidence_coordinator = _build_evidence_coordinator(settings)
    walter_agent = _build_walter_agent(settings)
    house_agent = _build_house_agent(settings)
    harvey_agent = _build_harvey_agent(settings)

    return ExecutiveWorkflow(
        prediction_client=_build_prediction_client(settings),
        evidence_coordinator=evidence_coordinator,
        walter_agent=walter_agent,
        house_agent=house_agent,
        harvey_agent=harvey_agent,
        ceo_synthesizer=CEOSynthesizer(),
        retmax=settings.retmax,
        top_k=settings.top_k,
    )


def _build_prediction_client(settings: AppSettings) -> TxGemmaClient | StubPredictionClient:
    if settings.use_stub_predictions or not settings.txgemma_endpoint_name:
        return StubPredictionClient()
    return build_txgemma_client(
        endpoint_name=settings.txgemma_endpoint_name,
        region_name=settings.txgemma_region_name,
    )


def _build_router_agent(settings: AppSettings) -> ConsultRouterAgent:
    if not settings.router_model_id:
        return ConsultRouterAgent()
    return ConsultRouterAgent.from_bedrock(
        model_id=settings.router_model_id,
        region_name=settings.bedrock_region_name,
    )


def _build_walter_agent(settings: AppSettings) -> WalterAgent:
    if not settings.walter_agent_model_id:
        return WalterAgent()
    return build_bedrock_walter_agent(
        model_id=settings.walter_agent_model_id,
        region_name=settings.bedrock_region_name,
    )


def _build_house_agent(settings: AppSettings) -> HouseAgent:
    if not settings.house_agent_model_id:
        return HouseAgent()
    return build_bedrock_house_agent(
        model_id=settings.house_agent_model_id,
        region_name=settings.bedrock_region_name,
    )


def _build_harvey_agent(settings: AppSettings) -> HarveyAgent:
    if not settings.harvey_agent_model_id:
        return HarveyAgent()
    return build_bedrock_harvey_agent(
        model_id=settings.harvey_agent_model_id,
        region_name=settings.bedrock_region_name,
    )


def _build_evidence_coordinator(settings: AppSettings) -> EvidenceCoordinator:
    pubmed_client = PubMedClient(
        tool=settings.tool_name,
        email=settings.contact_email,
        api_key=settings.pubmed_api_key,
        timeout=settings.http_timeout_seconds,
        cache_ttl_seconds=settings.cache_ttl_seconds,
    )
    return EvidenceCoordinator(
        pubmed_client=pubmed_client,
        pubmed_query_planner=_build_pubmed_query_planner(settings, pubmed_client),
        pubchem_client=PubChemClient(
            tool=settings.tool_name,
            email=settings.contact_email,
            timeout=settings.http_timeout_seconds,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        ),
        chembl_client=ChEMBLClient(
            tool=settings.tool_name,
            email=settings.contact_email,
            timeout=settings.http_timeout_seconds,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        ),
        clinicaltrials_client=ClinicalTrialsClient(
            tool=settings.tool_name,
            email=settings.contact_email,
            timeout=settings.http_timeout_seconds,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        ),
        openfda_client=OpenFDAClient(
            tool=settings.tool_name,
            email=settings.contact_email,
            timeout=settings.http_timeout_seconds,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        ),
    )


def _build_pubmed_query_planner(settings: AppSettings, pubmed_client: PubMedClient):
    if not settings.pubmed_query_planner_model_id:
        return None
    return build_bedrock_pubmed_query_agent(
        pubmed_client,
        model_id=settings.pubmed_query_planner_model_id,
        region_name=settings.bedrock_region_name,
        retmax=settings.retmax,
    )
