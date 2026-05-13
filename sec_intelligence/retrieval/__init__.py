"""
Layer 4: Retrieval Layer
Hybrid retrieval (BM25 + Vector), reranking, compression
"""
import logging
from typing import List, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

from sec_intelligence.config import get_config
from sec_intelligence.schemas import (
    StructuredChunk, RetrievalContext, RetrievedChunk, RetrievalResult, FilingType
)
from sec_intelligence.storage import get_postgres_db, get_vector_store
from sec_intelligence.storage.models import ChunkRecord

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 keyword-based retrieval"""
    
    def __init__(self):
        self.config = get_config()
        self.db = get_postgres_db()
        self.bm25_index = None
        self.chunks_by_id = {}
    
    def build_index(self):
        """Build BM25 index from all chunks"""
        try:
            session = self.db.get_session()
            chunks = session.query(ChunkRecord).all()
            
            # Tokenize chunks
            tokenized = [chunk.text.lower().split() for chunk in chunks]
            self.bm25_index = BM25Okapi(tokenized)
            
            # Store mapping
            self.chunks_by_id = {chunk.id: chunk for chunk in chunks}
            
            logger.info(f"Built BM25 index with {len(chunks)} chunks")
            session.close()
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
    
    def search(self, query: str, k: int = 10) -> List[tuple]:
        """
        Search with BM25
        Returns: List of (chunk_id, score) tuples
        """
        try:
            tokens = query.lower().split()
            scores = self.bm25_index.get_scores(tokens)
            
            # Get top k with scores
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    chunk_id = list(self.chunks_by_id.keys())[idx]
                    results.append((chunk_id, float(scores[idx])))
            
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []


class VectorRetriever:
    """Vector-based semantic retrieval"""
    
    def __init__(self):
        self.config = get_config()
        self.embedding_model = SentenceTransformer(
            self.config.vector_store.embedding_model
        )
        self.vector_store = get_vector_store()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return []
    
    def search(self, query: str, k: int = 10) -> List[tuple]:
        """
        Search vector store
        Returns: List of (chunk_id, score) tuples
        """
        try:
            query_embedding = self.embed_text(query)
            
            results = self.vector_store.search_vectors(
                collection_name=self.config.vector_store.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=0.0
            )
            
            # Extract IDs and scores
            return [(str(r.id), r.score) for r in results]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []


class Reranker:
    """Cross-encoder based reranking"""
    
    def __init__(self):
        self.config = get_config()
        self.reranker = CrossEncoder(
            self.config.retrieval.reranker_model
        )
    
    def rerank(self, query: str, chunks: List[StructuredChunk], k: int = 5) -> List[tuple]:
        """
        Rerank chunks by relevance
        
        Returns: List of (chunk, score) tuples
        """
        try:
            if not chunks:
                return []
            
            # Create pairs
            pairs = [[query, chunk.text] for chunk in chunks]
            
            # Score with cross-encoder
            scores = self.reranker.predict(pairs)
            
            # Sort by score descending
            ranked = sorted(
                zip(chunks, scores),
                key=lambda x: x[1],
                reverse=True
            )[:k]
            
            return ranked
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return []


class ChunkCompressor:
    """Compress chunks before passing to LLM"""
    
    @staticmethod
    def compress_chunk(chunk: StructuredChunk, max_tokens: int = 300) -> str:
        """
        Compress chunk to key information
        Simple approach: take first max_tokens chars
        """
        text = chunk.text
        
        # For now, simple truncation
        if len(text) > max_tokens * 4:  # Rough estimate: 4 chars per token
            text = text[:max_tokens * 4] + "..."
        
        return text


class HybridRetriever:
    """
    Combines BM25 and vector search
    """
    
    def __init__(self):
        self.config = get_config()
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()
        self.reranker = Reranker()
        self.compressor = ChunkCompressor()
        self.db = get_postgres_db()
    
    def initialize(self):
        """Initialize retriever indices"""
        self.bm25_retriever.build_index()
        logger.info("Hybrid retriever initialized")
    
    def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        k_initial: Optional[int] = None,
        k_final: Optional[int] = None
    ) -> RetrievalResult:
        """
        Hybrid retrieval with filtering, reranking, compression
        """
        try:
            k_initial = k_initial or self.config.retrieval.initial_retrieval_k
            k_final = k_final or self.config.retrieval.reranked_k
            
            # Step 1: Metadata filtering (if applicable)
            filters = self._build_filters(context)
            
            # Step 2: Hybrid search
            bm25_results = self.bm25_retriever.search(query, k=k_initial)
            vector_results = self.vector_retriever.search(query, k=k_initial)
            
            # Combine scores (weighted average)
            combined = self._combine_results(
                bm25_results,
                vector_results,
                self.config.retrieval.bm25_weight,
                self.config.retrieval.vector_weight
            )
            
            # Fetch chunks from database
            chunk_ids = [chunk_id for chunk_id, _ in combined]
            chunks = self._fetch_chunks(chunk_ids)
            
            # Step 3: Reranking
            if self.config.retrieval.use_reranking and len(chunks) > k_final:
                reranked = self.reranker.rerank(query, chunks, k=k_final)
                retrieved_chunks = [
                    RetrievedChunk(
                        chunk=chunk,
                        relevance_score=float(score),
                        retrieval_method="hybrid",
                        reranked=True,
                        rerank_score=float(score)
                    )
                    for chunk, score in reranked
                ]
            else:
                retrieved_chunks = [
                    RetrievedChunk(
                        chunk=chunk,
                        relevance_score=0.5,
                        retrieval_method="hybrid",
                        reranked=False
                    )
                    for chunk in chunks[:k_final]
                ]
            
            # Step 4: Compression
            for retrieved in retrieved_chunks:
                compressed = self.compressor.compress_chunk(retrieved.chunk)
                retrieved.chunk.text = compressed
            
            return RetrievalResult(
                query=query,
                context=context,
                chunks=retrieved_chunks,
                total_retrieved=len(retrieved_chunks),
                retrieval_time_ms=0.0  # TODO: add timing
            )
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return RetrievalResult(
                query=query,
                context=context,
                chunks=[],
                total_retrieved=0,
                retrieval_time_ms=0.0
            )
    
    def _build_filters(self, context: RetrievalContext) -> dict:
        """Build database filters from context"""
        filters = {}
        
        if context.ticker:
            filters['ticker'] = context.ticker
        if context.filing_types:
            filters['filing_types'] = [ft.value for ft in context.filing_types]
        if context.sections:
            filters['sections'] = context.sections
        
        return filters
    
    def _combine_results(
        self,
        bm25_results: List[tuple],
        vector_results: List[tuple],
        bm25_weight: float,
        vector_weight: float
    ) -> List[tuple]:
        """Combine and normalize results"""
        scores = {}
        
        # Normalize and add BM25 scores
        if bm25_results:
            max_bm25 = max([score for _, score in bm25_results]) if bm25_results else 1
            for chunk_id, score in bm25_results:
                scores[chunk_id] = (score / max_bm25) * bm25_weight
        
        # Normalize and add vector scores
        if vector_results:
            max_vector = max([score for _, score in vector_results]) if vector_results else 1
            for chunk_id, score in vector_results:
                if chunk_id in scores:
                    scores[chunk_id] += (score / max_vector) * vector_weight
                else:
                    scores[chunk_id] = (score / max_vector) * vector_weight
        
        # Sort by combined score
        combined = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return combined
    
    def _fetch_chunks(self, chunk_ids: List[str]) -> List[StructuredChunk]:
        """Fetch chunks from database"""
        try:
            session = self.db.get_session()
            chunk_records = session.query(ChunkRecord).filter(
                ChunkRecord.id.in_(chunk_ids)
            ).all()
            
            chunks = [
                StructuredChunk(
                    chunk_id=record.id,
                    filing_id=record.filing_id,
                    section=record.section,
                    subsection=record.subsection,
                    text=record.text,
                    chunk_type=record.chunk_type,
                    ticker=record.ticker,
                    filing_type=FilingType(record.filing_type),
                    filing_date=record.filing_date,
                    fiscal_year=record.fiscal_year,
                    heading=record.heading,
                    is_table=record.is_table,
                    page_number=record.page_number,
                    byte_offset=record.byte_offset
                )
                for record in chunk_records
            ]
            
            session.close()
            return chunks
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}")
            return []


class RetrievalAgent:
    """
    Layer 4: Retrieval Agent
    Decides retrieval strategy and manages retrieval operations
    """
    
    def __init__(self):
        self.config = get_config()
        self.retriever = HybridRetriever()
        self.retriever.initialize()
    
    def retrieve_for_query(self, query: str, context: RetrievalContext) -> RetrievalResult:
        """
        Main entry point for retrieval
        Decides whether to use structured DB or RAG
        """
        logger.info(f"Retrieving for query: {query}")
        
        # Check if query can be answered from structured data
        if context.risk_types:
            # Risk query - can use structured risks table
            logger.debug("Risk query detected - could use structured retrieval")
        
        # For now, use hybrid retrieval
        result = self.retriever.retrieve(query, context)
        
        logger.info(f"Retrieved {result.total_retrieved} chunks")
        return result


# Singleton instance
_retrieval_agent: Optional[RetrievalAgent] = None


def get_retrieval_agent() -> RetrievalAgent:
    """Get retrieval agent instance"""
    global _retrieval_agent
    if _retrieval_agent is None:
        _retrieval_agent = RetrievalAgent()
    return _retrieval_agent
