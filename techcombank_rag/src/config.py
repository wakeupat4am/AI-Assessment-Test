from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parent

# Root configuration is the submission contract. The project-local file is
# retained only for backward compatibility with the original baseline.
load_dotenv(REPOSITORY_ROOT / ".env")
load_dotenv(APP_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _as_bool(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _as_optional_float(name: str) -> float | None:
    value = _env(name)
    return float(value) if value else None


def _as_path(name: str, default: Path) -> Path:
    value = Path(_env(name, str(default)))
    return value if value.is_absolute() else REPOSITORY_ROOT / value


@dataclass(frozen=True)
class Settings:
    app_root: Path = APP_ROOT
    repository_root: Path = REPOSITORY_ROOT
    index_dir: Path = field(
        default_factory=lambda: _as_path("INDEX_DIR", APP_ROOT / "data/index")
    )
    processed_dir: Path = field(
        default_factory=lambda: _as_path("PROCESSED_DIR", APP_ROOT / "data/processed")
    )
    evaluation_dir: Path = field(
        default_factory=lambda: _as_path("EVALUATION_DIR", APP_ROOT / "data/evaluation")
    )
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL"))
    embedding_device: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "cpu"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    evaluation_retrieval_k: int = field(
        default_factory=lambda: int(_env("EVALUATION_RETRIEVAL_K", "10"))
    )
    min_retrieval_score: float = field(
        default_factory=lambda: float(_env("MIN_RETRIEVAL_SCORE", "0.50"))
    )
    b2_enabled: bool = field(default_factory=lambda: _as_bool("B2_ENABLED"))
    enable_auto_search: bool = field(
        default_factory=lambda: _as_bool("ENABLE_AUTO_SEARCH")
    )
    enable_page_dedup: bool = field(
        default_factory=lambda: _as_bool("ENABLE_PAGE_DEDUP")
    )
    enable_diversity_selection: bool = field(
        default_factory=lambda: _as_bool("ENABLE_DIVERSITY_SELECTION")
    )
    b2_initial_top_k: int = field(
        default_factory=lambda: int(_env("B2_INITIAL_TOP_K", "20"))
    )
    b2_second_round_top_k: int = field(
        default_factory=lambda: int(_env("B2_SECOND_ROUND_TOP_K", "12"))
    )
    b2_max_search_rounds: int = field(
        default_factory=lambda: int(_env("B2_MAX_SEARCH_ROUNDS", "2"))
    )
    b2_max_per_page: int = field(
        default_factory=lambda: int(_env("B2_MAX_PER_PAGE", "2"))
    )
    b2_page_similarity_threshold: float = field(
        default_factory=lambda: float(_env("B2_PAGE_SIMILARITY_THRESHOLD", "0.88"))
    )
    b2_final_k: int = field(
        default_factory=lambda: int(_env("B2_FINAL_K", "6"))
    )
    b2_max_evidence_tokens: int = field(
        default_factory=lambda: int(_env("B2_MAX_EVIDENCE_TOKENS", "5000"))
    )
    b2_diversity_lambda: float = field(
        default_factory=lambda: float(_env("B2_DIVERSITY_LAMBDA", "0.70"))
    )
    b2_log_path: Path = field(
        default_factory=lambda: _as_path(
            "B2_LOG_PATH", APP_ROOT / "data/evaluation/router/b2-search.jsonl"
        )
    )
    query_embedding_cache_size: int = field(
        default_factory=lambda: int(_env("QUERY_EMBEDDING_CACHE_SIZE", "128"))
    )
    xrouter_enabled: bool = field(
        default_factory=lambda: _as_bool("XROUTER_ENABLED")
    )
    xrouter_type: str = field(
        default_factory=lambda: _env("XROUTER_TYPE", "rule_based")
    )
    xrouter_config_path: Path = field(
        default_factory=lambda: _as_path(
            "XROUTER_CONFIG", APP_ROOT / "config/xrouter_b1a.json"
        )
    )
    xrouter_log_path: Path = field(
        default_factory=lambda: _as_path(
            "XROUTER_LOG_PATH", APP_ROOT / "data/evaluation/router/b1a-router.jsonl"
        )
    )
    xrouter_confidence_threshold: float | None = field(
        default_factory=lambda: _as_optional_float("XROUTER_CONFIDENCE_THRESHOLD")
    )
    router_llm_provider: str = field(
        default_factory=lambda: _env(
            "ROUTER_LLM_PROVIDER", _env("LLM_PROVIDER", "openai_compatible")
        )
    )
    router_llm_base_url: str = field(
        default_factory=lambda: _env("ROUTER_LLM_BASE_URL", _env("LLM_BASE_URL"))
    )
    router_llm_model: str = field(
        default_factory=lambda: _env(
            "ROUTER_LLM_MODEL",
            _env("LLM_MODEL_FAST", _env("LLM_MODEL", _env("LLM_MODEL_STRONG"))),
        )
    )
    router_llm_api_key: str = field(
        default_factory=lambda: _env("ROUTER_LLM_API_KEY", _env("LLM_API_KEY"))
    )
    router_llm_temperature: float = field(
        default_factory=lambda: float(_env("ROUTER_LLM_TEMPERATURE", "0"))
    )
    router_llm_timeout_seconds: float = field(
        default_factory=lambda: float(_env("ROUTER_LLM_TIMEOUT_SECONDS", "60"))
    )
    router_llm_max_new_tokens: int = field(
        default_factory=lambda: int(_env("ROUTER_LLM_MAX_NEW_TOKENS", "128"))
    )
    narrative_index_dir: Path = field(
        default_factory=lambda: _as_path(
            "NARRATIVE_INDEX_DIR", APP_ROOT / "data/index/b1a_narrative_index"
        )
    )
    structured_index_dir: Path = field(
        default_factory=lambda: _as_path(
            "STRUCTURED_INDEX_DIR",
            APP_ROOT / "data/index/a2_paddleocr_vl_1_6_layout_aware",
        )
    )
    multi_repr_index_dir: Path = field(
        default_factory=lambda: _as_path(
            "MULTI_REPR_INDEX_DIR",
            APP_ROOT / "data/index/a3_paddleocr_vl_1_6_multigranularity",
        )
    )
    a31_index_root: Path = field(
        default_factory=lambda: _as_path(
            "A31_INDEX_ROOT", APP_ROOT / "data/index/a31_semantic_multirepr"
        )
    )
    a31_parent_characters: int = field(
        default_factory=lambda: int(_env("A31_PARENT_CHARACTERS", "1800"))
    )
    hyde_enabled: bool = field(default_factory=lambda: _as_bool("ENABLE_HYDE"))
    hyde_mode: str = field(default_factory=lambda: _env("HYDE_MODE", "fusion"))
    hyde_prompt_variant: str = field(
        default_factory=lambda: _env("HYDE_PROMPT_VARIANT", "finance_safe")
    )
    hyde_config_path: Path = field(
        default_factory=lambda: _as_path(
            "HYDE_CONFIG", APP_ROOT / "config/hyde.json"
        )
    )
    hyde_llm_provider: str = field(
        default_factory=lambda: _env("HYDE_PROVIDER", _env("LLM_PROVIDER", "openai_compatible"))
    )
    hyde_llm_base_url: str = field(
        default_factory=lambda: _env("HYDE_BASE_URL", _env("LLM_BASE_URL"))
    )
    hyde_llm_model: str = field(
        default_factory=lambda: _env(
            "HYDE_MODEL", _env("LLM_MODEL_FAST", _env("LLM_MODEL", _env("LLM_MODEL_STRONG")))
        )
    )
    hyde_llm_api_key: str = field(
        default_factory=lambda: _env("HYDE_API_KEY", _env("LLM_API_KEY"))
    )
    hyde_llm_temperature: float = field(
        default_factory=lambda: float(_env("HYDE_TEMPERATURE", "0"))
    )
    hyde_llm_timeout_seconds: float = field(
        default_factory=lambda: float(_env("HYDE_TIMEOUT_SECONDS", "60"))
    )
    hyde_llm_max_new_tokens: int = field(
        default_factory=lambda: int(_env("HYDE_MAX_TOKENS", "180"))
    )
    hyde_candidate_k: int = field(
        default_factory=lambda: int(_env("HYDE_CANDIDATE_K", "20"))
    )
    hyde_rrf_k: int = field(
        default_factory=lambda: int(_env("HYDE_RRF_K", "60"))
    )
    hyde_original_weight: float = field(
        default_factory=lambda: float(_env("HYDE_ORIGINAL_WEIGHT", "0.5"))
    )
    hyde_hypothetical_weight: float = field(
        default_factory=lambda: float(_env("HYDE_HYPOTHETICAL_WEIGHT", "0.5"))
    )
    hyde_cache_size: int = field(
        default_factory=lambda: int(_env("HYDE_CACHE_SIZE", "128"))
    )
    hyde_log_path: Path = field(
        default_factory=lambda: _as_path(
            "HYDE_LOG_PATH", APP_ROOT / "data/evaluation/router/hyde.jsonl"
        )
    )

    b4_retrieval_mode: str = field(
        default_factory=lambda: _env("B4_RETRIEVAL_MODE", "off")
    )
    b4_config_path: Path = field(
        default_factory=lambda: _as_path(
            "B4_CONFIG", APP_ROOT / "config/b4_retrieval.json"
        )
    )
    b4_bm25_artifact_path: Path = field(
        default_factory=lambda: _as_path(
            "B4_BM25_ARTIFACT",
            APP_ROOT / "data/index/a31_semantic_multirepr/all/bm25.json.gz",
        )
    )
    b4_frozen_index_dir: Path = field(
        default_factory=lambda: _as_path(
            "B4_FROZEN_INDEX_DIR",
            APP_ROOT / "data/index/a31_semantic_multirepr/all",
        )
    )
    b4_frozen_config_path: Path = field(
        default_factory=lambda: _as_path(
            "B4_FROZEN_CONFIG", APP_ROOT / "config/b4_retrieval.json"
        )
    )
    b4_cross_encoder_model: str = field(
        default_factory=lambda: _env(
            "B4_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
    )
    b4_cross_encoder_device: str = field(
        default_factory=lambda: _env("B4_CROSS_ENCODER_DEVICE", "cpu")
    )
    b4_cross_encoder_batch_size: int = field(
        default_factory=lambda: int(_env("B4_CROSS_ENCODER_BATCH_SIZE", "8"))
    )
    b4_cross_encoder_max_length: int = field(
        default_factory=lambda: int(_env("B4_CROSS_ENCODER_MAX_LENGTH", "512"))
    )
    b4_cross_encoder_threads: int = field(
        default_factory=lambda: int(_env("B4_CROSS_ENCODER_THREADS", "8"))
    )
    b5_agent_enabled: bool = field(default_factory=lambda: _as_bool("B5_AGENT_ENABLED"))
    b5_agent_config_path: Path = field(
        default_factory=lambda: _as_path(
            "B5_AGENT_CONFIG", APP_ROOT / "config/b5_tool_agent.json"
        )
    )
    b5_agent_log_path: Path = field(
        default_factory=lambda: _as_path(
            "B5_AGENT_LOG_PATH", APP_ROOT / "data/evaluation/router/b5-agent.jsonl"
        )
    )

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai_compatible"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    llm_model_fast: str = field(default_factory=lambda: _env("LLM_MODEL_FAST"))
    llm_model_strong: str = field(default_factory=lambda: _env("LLM_MODEL_STRONG"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0")))
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(_env("LLM_TIMEOUT_SECONDS", "300"))
    )
    max_new_tokens: int = field(default_factory=lambda: int(_env("MAX_NEW_TOKENS", "256")))
    max_history_turns: int = field(
        default_factory=lambda: int(_env("MAX_HISTORY_TURNS", "4"))
    )
    max_search_steps: int = field(
        default_factory=lambda: int(_env("MAX_SEARCH_STEPS", "3"))
    )

    input_price_per_million: float | None = field(
        default_factory=lambda: _as_optional_float("LLM_INPUT_PRICE_PER_MILLION")
    )
    cached_input_price_per_million: float | None = field(
        default_factory=lambda: _as_optional_float("LLM_CACHED_INPUT_PRICE_PER_MILLION")
    )
    cache_write_price_per_million: float | None = field(
        default_factory=lambda: _as_optional_float("LLM_CACHE_WRITE_PRICE_PER_MILLION")
    )
    output_price_per_million: float | None = field(
        default_factory=lambda: _as_optional_float("LLM_OUTPUT_PRICE_PER_MILLION")
    )
    debug: bool = field(default_factory=lambda: _as_bool("DEBUG"))

    @property
    def fast_model(self) -> str:
        return self.llm_model_fast or self.llm_model or self.llm_model_strong

    @property
    def strong_model(self) -> str:
        return self.llm_model_strong or self.llm_model or self.llm_model_fast

    @property
    def default_model(self) -> str:
        return self.llm_model or self.llm_model_strong or self.llm_model_fast

    def model_for_purpose(self, purpose: str) -> str:
        """Use the strong model for finalization and the fast model elsewhere.

        If only one model is configured both properties resolve to it, which is
        the documented single-model fallback used by clean-machine graders.
        """
        if purpose in {"answer", "repair"}:
            return self.strong_model
        return self.fast_model


def get_settings() -> Settings:
    return Settings()
