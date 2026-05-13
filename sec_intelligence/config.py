from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import Literal, Optional
from pathlib import Path


class SECConfig(BaseModel):
    """SEC filing configuration"""
    sec_api_base: str = "https://www.sec.gov/cgi-bin/browse-edgar"
    sec_filings_api: str = "https://www.sec.gov/Archives"
    filing_types: list[str] = ["10-K", "10-Q", "8-K", "13D", "13G", "DEF 14A"]
    sec_rate_limit_delay: float = 0.1


class DatabaseConfig(BaseModel):
    """PostgreSQL configuration"""
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sec_intelligence"
    db_user: str = "sec_user"
    db_password: str = "sec_password"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False
    
    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg2://"
            f"{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )


class VectorStoreConfig(BaseModel):
    """Qdrant vector store configuration"""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    vectorstore_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vectorstore_embedding_dim: int = 384
    vectorstore_collection_name: str = "sec_filings"

    @property
    def embedding_model(self) -> str:
        return self.vectorstore_embedding_model

    @property
    def embedding_dim(self) -> int:
        return self.vectorstore_embedding_dim

    @property
    def collection_name(self) -> str:
        return self.vectorstore_collection_name


class ObjectStorageConfig(BaseModel):
    """Local object storage configuration"""
    storage_base_path: Path = Path("/tmp/sec_storage")
    storage_raw_filings_path: str = "raw_filings"
    storage_processed_filings_path: str = "processed_filings"
    
    @property
    def base_path(self) -> Path:
        return self.storage_base_path
    
    @property
    def raw_filings_path(self) -> str:
        return self.storage_raw_filings_path
    
    @property
    def processed_filings_path(self) -> str:
        return self.storage_processed_filings_path


class LLMConfig(BaseSettings):
    """LLM configuration"""
    llm_provider: Literal["anthropic", "openai"] = "openai"
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    @property
    def provider(self) -> str:
        return self.llm_provider
    
    @property
    def model_name(self) -> str:
        return self.llm_model_name
    
    @property
    def api_key(self) -> Optional[str]:
        return self.llm_api_key


class AgentConfig(BaseModel):
    """Agent configuration"""
    agent_max_retries: int = 3
    agent_timeout_seconds: int = 300
    agent_batch_size: int = 10


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    retrieval_bm25_weight: float = 0.3
    retrieval_vector_weight: float = 0.7
    retrieval_initial_retrieval_k: int = 30
    retrieval_reranked_k: int = 5
    retrieval_use_reranking: bool = True
    retrieval_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    @property
    def bm25_weight(self) -> float:
        return self.retrieval_bm25_weight

    @property
    def vector_weight(self) -> float:
        return self.retrieval_vector_weight

    @property
    def initial_retrieval_k(self) -> int:
        return self.retrieval_initial_retrieval_k

    @property
    def reranked_k(self) -> int:
        return self.retrieval_reranked_k

    @property
    def use_reranking(self) -> bool:
        return self.retrieval_use_reranking

    @property
    def reranker_model(self) -> str:
        return self.retrieval_reranker_model


class AppConfig(BaseSettings):
    """Main application configuration"""
    app_name: str = "SEC Intelligence Platform"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Sub-configs
    sec: SECConfig = SECConfig()
    database: DatabaseConfig = DatabaseConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    object_storage: ObjectStorageConfig = ObjectStorageConfig()
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )


# Singleton config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create config instance"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config

