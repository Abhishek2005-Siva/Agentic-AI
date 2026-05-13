"""
Storage layer - manages PostgreSQL and Vector DB
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import Optional
from pathlib import Path
import logging

from sec_intelligence.config import get_config
from sec_intelligence.storage.models import Base

logger = logging.getLogger(__name__)


class PostgresDB:
    """PostgreSQL database connection and session management"""
    
    def __init__(self):
        self.config = get_config()
        self.engine = None
        self.SessionLocal = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(
                self.config.database.db_url,
                poolclass=QueuePool,
                pool_size=self.config.database.db_pool_size,
                max_overflow=self.config.database.db_max_overflow,
                echo=self.config.database.db_echo,
                pool_pre_ping=True,  # Verify connection before use
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("PostgreSQL connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    def init_db(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("PostgreSQL connection closed")


class VectorStore:
    """Qdrant vector store management"""
    
    def __init__(self):
        self.config = get_config()
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize vector store connection"""
        try:
            self.client = QdrantClient(
                url=self.config.vector_store.qdrant_url,
                api_key=self.config.vector_store.qdrant_api_key,
            )
            
            # Test connection
            try:
                self.client.get_collections()
                logger.info(f"Connected to Qdrant at {self.config.vector_store.qdrant_url}")
            except Exception:
                logger.warning("Qdrant not available, using in-memory mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def create_collection(self, collection_name: Optional[str] = None):
        """Create vector collection if not exists"""
        collection_name = collection_name or self.config.vector_store.collection_name
        
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_store.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created vector collection: {collection_name}")
            else:
                logger.info(f"Vector collection exists: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_vectors(
        self,
        collection_name: str,
        points: list[PointStruct]
    ):
        """Add vectors to collection"""
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.debug(f"Added {len(points)} vectors to {collection_name}")
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise
    
    def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> list:
        """Search vector store"""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise
    
    def delete_collection(self, collection_name: str):
        """Delete collection"""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def close(self):
        """Close vector store connection"""
        if self.client:
            logger.info("Vector store connection closed")


class ObjectStorage:
    """Local file storage for raw and processed filings"""
    
    def __init__(self):
        self.config = get_config()
        self.base_path = self.config.object_storage.storage_base_path
        self._initialize()
    
    def _initialize(self):
        """Create storage directories"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            (self.base_path / self.config.object_storage.raw_filings_path).mkdir(
                parents=True, exist_ok=True
            )
            (self.base_path / self.config.object_storage.processed_filings_path).mkdir(
                parents=True, exist_ok=True
            )
            logger.info(f"Object storage initialized at {self.base_path}")
        except Exception as e:
            logger.error(f"Failed to initialize object storage: {e}")
            raise
    
    def get_raw_path(self, accession_number: str) -> Path:
        """Get path for raw filing"""
        return (
            self.base_path / 
            self.config.object_storage.raw_filings_path /
            f"{accession_number}.html"
        )
    
    def get_processed_path(self, accession_number: str) -> Path:
        """Get path for processed filing"""
        return (
            self.base_path /
            self.config.object_storage.processed_filings_path /
            f"{accession_number}.json"
        )
    
    def save_raw(self, accession_number: str, content: str) -> Path:
        """Save raw filing content"""
        try:
            path = self.get_raw_path(accession_number)
            path.write_text(content, encoding='utf-8')
            logger.debug(f"Saved raw filing: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save raw filing: {e}")
            raise
    
    def save_processed(self, accession_number: str, content: str) -> Path:
        """Save processed filing content"""
        try:
            path = self.get_processed_path(accession_number)
            path.write_text(content, encoding='utf-8')
            logger.debug(f"Saved processed filing: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save processed filing: {e}")
            raise
    
    def read_raw(self, accession_number: str) -> str:
        """Read raw filing content"""
        try:
            path = self.get_raw_path(accession_number)
            return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read raw filing: {e}")
            raise
    
    def read_processed(self, accession_number: str) -> str:
        """Read processed filing content"""
        try:
            path = self.get_processed_path(accession_number)
            return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read processed filing: {e}")
            raise


# Singleton instances
_postgres_db: Optional[PostgresDB] = None
_vector_store: Optional[VectorStore] = None
_object_storage: Optional[ObjectStorage] = None


def get_postgres_db() -> PostgresDB:
    """Get PostgreSQL database instance"""
    global _postgres_db
    if _postgres_db is None:
        _postgres_db = PostgresDB()
    return _postgres_db


def get_vector_store() -> VectorStore:
    """Get vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_object_storage() -> ObjectStorage:
    """Get object storage instance"""
    global _object_storage
    if _object_storage is None:
        _object_storage = ObjectStorage()
    return _object_storage


def init_all_storage():
    """Initialize all storage layers"""
    postgres = get_postgres_db()
    postgres.init_db()
    
    vector_store = get_vector_store()
    vector_store.create_collection()
    
    get_object_storage()
    
    logger.info("All storage layers initialized")
