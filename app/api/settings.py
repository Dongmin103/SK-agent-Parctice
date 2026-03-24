from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "on"}
_ENV_ALIAS_GROUPS = (
    ("APP_TOOL_NAME", "EUTILS_TOOL"),
    ("APP_CONTACT_EMAIL", "EUTILS_EMAIL"),
    ("TXGEMMA_SAGEMAKER_ENDPOINT_NAME", "TXGEMMA_ENDPOINT_NAME"),
    ("TXGEMMA_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
    ("BEDROCK_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
)


def _get_first_env(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None


def _get_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_VALUES


def _get_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = env.get(name)
    if value is None:
        return default
    return float(value)


def _get_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    return int(value)


def _read_dotenv_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    resolved: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue

        name, value = line.split("=", 1)
        key = name.strip()
        if not key:
            continue

        parsed_value = value.strip()
        if (
            len(parsed_value) >= 2
            and parsed_value[0] == parsed_value[-1]
            and parsed_value[0] in {"'", '"'}
        ):
            parsed_value = parsed_value[1:-1]

        resolved[key] = parsed_value

    return resolved


def _load_file_env(cwd: str | os.PathLike[str] | None = None) -> dict[str, str]:
    base_dir = Path(cwd) if cwd is not None else Path.cwd()
    resolved: dict[str, str] = {}
    for filename in (".env", ".env.local"):
        resolved.update(_read_dotenv_file(base_dir / filename))
    return resolved


def _overlay_env(base: dict[str, str], override: Mapping[str, str]) -> dict[str, str]:
    resolved = dict(base)
    for group in _ENV_ALIAS_GROUPS:
        if any(name in override for name in group):
            for name in group:
                resolved.pop(name, None)
    resolved.update(override)
    return resolved


@dataclass(slots=True)
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    use_stub_workflows: bool = False
    tool_name: str = "agentic-ai-poc"
    contact_email: str = "local@example.com"
    pubmed_api_key: str | None = None
    http_timeout_seconds: float = 10.0
    cache_ttl_seconds: int = 86_400
    retmax: int = 10
    top_k: int = 5
    use_stub_predictions: bool = False
    txgemma_endpoint_name: str | None = None
    txgemma_region_name: str | None = None
    bedrock_region_name: str | None = None
    pubmed_query_planner_model_id: str | None = None
    router_model_id: str | None = None
    walter_agent_model_id: str | None = None
    house_agent_model_id: str | None = None
    harvey_agent_model_id: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AppSettings":
        resolved_env = dict(os.environ if env is None else env)
        return cls(
            host=resolved_env.get("API_HOST", "127.0.0.1"),
            port=_get_int(resolved_env, "API_PORT", 8000),
            use_stub_workflows=_get_bool(resolved_env, "API_USE_STUB_WORKFLOWS", False),
            tool_name=resolved_env.get("APP_TOOL_NAME")
            or resolved_env.get("EUTILS_TOOL")
            or "agentic-ai-poc",
            contact_email=resolved_env.get("APP_CONTACT_EMAIL")
            or resolved_env.get("EUTILS_EMAIL")
            or "local@example.com",
            pubmed_api_key=resolved_env.get("NCBI_API_KEY"),
            http_timeout_seconds=_get_float(resolved_env, "APP_HTTP_TIMEOUT_SECONDS", 10.0),
            cache_ttl_seconds=_get_int(resolved_env, "APP_CACHE_TTL_SECONDS", 86_400),
            retmax=_get_int(resolved_env, "APP_EVIDENCE_RETMAX", 10),
            top_k=_get_int(resolved_env, "APP_EVIDENCE_TOP_K", 5),
            use_stub_predictions=_get_bool(resolved_env, "APP_USE_STUB_PREDICTIONS", False),
            txgemma_endpoint_name=_get_first_env(
                resolved_env,
                "TXGEMMA_SAGEMAKER_ENDPOINT_NAME",
                "TXGEMMA_ENDPOINT_NAME",
            ),
            txgemma_region_name=_get_first_env(
                resolved_env,
                "TXGEMMA_AWS_REGION",
                "AWS_REGION",
                "AWS_DEFAULT_REGION",
            ),
            bedrock_region_name=_get_first_env(
                resolved_env,
                "BEDROCK_AWS_REGION",
                "AWS_REGION",
                "AWS_DEFAULT_REGION",
            ),
            pubmed_query_planner_model_id=resolved_env.get("BEDROCK_PUBMED_QUERY_MODEL_ID"),
            router_model_id=resolved_env.get("BEDROCK_ROUTER_MODEL_ID"),
            walter_agent_model_id=resolved_env.get("BEDROCK_WALTER_AGENT_MODEL_ID"),
            house_agent_model_id=resolved_env.get("BEDROCK_HOUSE_AGENT_MODEL_ID"),
            harvey_agent_model_id=resolved_env.get("BEDROCK_HARVEY_AGENT_MODEL_ID"),
        )

    @property
    def eutils_tool(self) -> str:
        return self.tool_name

    @property
    def eutils_email(self) -> str:
        return self.contact_email


def load_settings(
    env: Mapping[str, str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> AppSettings:
    resolved_env = _load_file_env(cwd)
    if env is None:
        resolved_env = _overlay_env(resolved_env, os.environ)
    else:
        resolved_env = _overlay_env(resolved_env, env)
    return AppSettings.from_env(resolved_env)
