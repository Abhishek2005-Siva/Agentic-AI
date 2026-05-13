"""
Database models for PostgreSQL storage
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, Float, JSON, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class FilingRecord(Base):
    """Raw filing record"""
    __tablename__ = "filings"
    
    id = Column(String(50), primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    filing_type = Column(String(20), nullable=False, index=True)
    cik = Column(String(20), nullable=False, index=True)
    filing_date = Column(DateTime, nullable=False, index=True)
    fiscal_year = Column(Integer, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    accession_number = Column(String(50), nullable=False, unique=True)
    url = Column(String(500), nullable=False)
    
    raw_text_path = Column(String(500))  # Path to raw storage
    raw_html_path = Column(String(500))
    source_hash = Column(String(64), nullable=False)
    
    ingested_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    structured_at = Column(DateTime)
    intelligence_extracted_at = Column(DateTime)
    
    filing_metadata = Column(JSON)  # Additional metadata
    
    chunks = relationship("ChunkRecord", back_populates="filing")
    intelligence = relationship("IntelligenceRecord", back_populates="filing", uselist=False)
    
    __table_args__ = (
        Index('idx_ticker_year', 'ticker', 'fiscal_year'),
        Index('idx_filing_type_date', 'filing_type', 'filing_date'),
    )


class ChunkRecord(Base):
    """Structured chunk record"""
    __tablename__ = "chunks"
    
    id = Column(String(100), primary_key=True)
    filing_id = Column(String(50), ForeignKey("filings.id"), nullable=False, index=True)
    
    section = Column(String(100), nullable=False, index=True)
    subsection = Column(String(255))
    text = Column(Text, nullable=False)
    chunk_type = Column(String(20), default="text")  # text, table, heading, bullet
    
    ticker = Column(String(10), nullable=False, index=True)
    filing_type = Column(String(20), nullable=False)
    filing_date = Column(DateTime, nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    
    heading = Column(String(500))
    is_table = Column(Boolean, default=False)
    table_data = Column(JSON)
    
    page_number = Column(Integer)
    byte_offset = Column(Integer)
    
    embedding_id = Column(String(50))  # Reference to vector store
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    filing = relationship("FilingRecord", back_populates="chunks")
    
    __table_args__ = (
        Index('idx_section_ticker', 'section', 'ticker'),
        Index('idx_chunk_filing', 'filing_id'),
    )


class IntelligenceRecord(Base):
    """Extracted intelligence record"""
    __tablename__ = "intelligence"
    
    id = Column(String(100), primary_key=True)
    filing_id = Column(String(50), ForeignKey("filings.id"), nullable=False, unique=True)
    
    executive_summary = Column(Text, nullable=False)
    key_highlights = Column(JSON)  # List of highlights
    
    extracted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    filing = relationship("FilingRecord", back_populates="intelligence")
    risks = relationship("RiskRecord", back_populates="intelligence")
    metrics = relationship("MetricRecord", back_populates="intelligence")
    entities = relationship("EntityRecord", back_populates="intelligence")
    events = relationship("EventRecord", back_populates="intelligence")


class RiskRecord(Base):
    """Extracted risk record"""
    __tablename__ = "risks"
    
    id = Column(String(100), primary_key=True)
    intelligence_id = Column(String(100), ForeignKey("intelligence.id"), nullable=False, index=True)
    
    risk_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)
    description = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    
    impact_area = Column(String(255))
    mitigation = Column(Text)
    
    related_chunks = Column(JSON)  # List of chunk IDs
    
    ticker = Column(String(10), nullable=False, index=True)
    fiscal_year = Column(Integer, nullable=False, index=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    intelligence = relationship("IntelligenceRecord", back_populates="risks")
    
    __table_args__ = (
        Index('idx_risk_type_ticker', 'risk_type', 'ticker'),
        Index('idx_severity_year', 'severity', 'fiscal_year'),
    )


class MetricRecord(Base):
    """Extracted financial metric record"""
    __tablename__ = "metrics"
    
    id = Column(String(100), primary_key=True)
    intelligence_id = Column(String(100), ForeignKey("intelligence.id"), nullable=False, index=True)
    
    metric_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(50), default="USD")
    period = Column(String(20), nullable=False)
    change_vs_prior = Column(Float)
    
    source_chunks = Column(JSON)
    
    ticker = Column(String(10), nullable=False, index=True)
    fiscal_year = Column(Integer, nullable=False, index=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    intelligence = relationship("IntelligenceRecord", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_metric_ticker_year', 'metric_name', 'ticker', 'fiscal_year'),
    )


class EntityRecord(Base):
    """Named entity record"""
    __tablename__ = "entities"
    
    id = Column(String(100), primary_key=True)
    intelligence_id = Column(String(100), ForeignKey("intelligence.id"), nullable=False, index=True)
    
    entity_type = Column(String(50), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    role = Column(String(255))
    description = Column(Text)
    
    related_chunks = Column(JSON)
    
    ticker = Column(String(10), nullable=False, index=True)
    fiscal_year = Column(Integer, nullable=False, index=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    intelligence = relationship("IntelligenceRecord", back_populates="entities")
    
    __table_args__ = (
        Index('idx_entity_name_ticker', 'name', 'ticker'),
    )


class EventRecord(Base):
    """Business event record"""
    __tablename__ = "events"
    
    id = Column(String(100), primary_key=True)
    intelligence_id = Column(String(100), ForeignKey("intelligence.id"), nullable=False, index=True)
    
    event_type = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    event_date = Column(DateTime)
    amount = Column(Float)
    counterparty = Column(String(255))
    
    related_chunks = Column(JSON)
    
    ticker = Column(String(10), nullable=False, index=True)
    fiscal_year = Column(Integer, nullable=False, index=True)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    intelligence = relationship("IntelligenceRecord", back_populates="events")
    
    __table_args__ = (
        Index('idx_event_type_ticker', 'event_type', 'ticker'),
    )


class RetrievalLog(Base):
    """Log of retrieval operations"""
    __tablename__ = "retrieval_logs"
    
    id = Column(String(100), primary_key=True)
    query = Column(String(500), nullable=False)
    ticker = Column(String(10))
    filing_types = Column(JSON)
    
    chunks_retrieved = Column(Integer)
    retrieval_method = Column(String(50))  # bm25, vector, hybrid
    execution_time_ms = Column(Float)
    
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
