"""
Data schemas for SEC Intelligence Platform
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any, Dict, List
from enum import Enum


class FilingType(str, Enum):
    """SEC Filing types"""
    K10 = "10-K"
    Q10 = "10-Q"
    K8 = "8-K"
    D13 = "13D"
    G13 = "13G"
    DEF14A = "DEF 14A"


class RiskType(str, Enum):
    """Types of risks extracted"""
    SUPPLY_CHAIN = "supply_chain"
    CYBERSECURITY = "cybersecurity"
    GEOPOLITICAL = "geopolitical"
    AI_REGULATION = "ai_regulation"
    LIQUIDITY = "liquidity"
    LITIGATION = "litigation"
    OPERATIONAL = "operational"


class RiskSeverity(str, Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# LAYER 1: INGESTION LAYER
# ============================================================================

class FilingMetadata(BaseModel):
    """Metadata for a filing"""
    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: str = Field(..., description="Full company name")
    filing_type: FilingType = Field(..., description="Type of filing")
    cik: str = Field(..., description="Central Index Key")
    filing_date: datetime = Field(..., description="Date filed with SEC")
    fiscal_year: int = Field(..., description="Fiscal year covered")
    period_end: datetime = Field(..., description="End of fiscal period")
    accession_number: str = Field(..., description="SEC accession number")
    url: str = Field(..., description="URL to SEC filing")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "filing_type": "10-K",
                "cik": "0001045810",
                "filing_date": "2025-02-27",
                "fiscal_year": 2025,
                "period_end": "2025-01-26",
                "accession_number": "0001047469-25-001234",
            }
        }


class RawFiling(BaseModel):
    """Raw filing from SEC"""
    metadata: FilingMetadata
    raw_text: str = Field(..., description="Raw HTML/text from SEC")
    raw_html: Optional[str] = Field(None, description="Raw HTML if available")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source_hash: str = Field(..., description="SHA256 hash of raw content")
    
    class Config:
        json_schema_extra = {
            "description": "Output of Ingestion Layer"
        }


# ============================================================================
# LAYER 2: STRUCTURING LAYER
# ============================================================================

class TableData(BaseModel):
    """Extracted table from filing"""
    table_type: str = Field(..., description="Type: balance_sheet, revenue, segment, etc")
    title: Optional[str] = None
    headers: List[str]
    rows: List[Dict[str, Any]]
    period: Optional[str] = None


class StructuredChunk(BaseModel):
    """Chunk after structuring"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    filing_id: str = Field(..., description="Reference to filing")
    section: str = Field(..., description="SEC section (Item 1, Item 1A, etc)")
    subsection: Optional[str] = Field(None, description="Subsection title")
    text: str = Field(..., description="Main text content")
    chunk_type: str = Field(default="text", description="text, table, heading, bullet")
    
    # Metadata
    ticker: str
    filing_type: FilingType
    filing_date: datetime
    fiscal_year: int
    
    # Structure preservation
    heading: Optional[str] = None
    is_table: bool = False
    table_data: Optional[TableData] = None
    
    # Position tracking
    page_number: Optional[int] = None
    byte_offset: Optional[int] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StructuredFiling(BaseModel):
    """Completely structured filing"""
    filing_id: str
    metadata: FilingMetadata
    chunks: List[StructuredChunk]
    section_map: Dict[str, List[str]] = Field(
        ..., description="Map of sections to chunk IDs"
    )
    tables: List[TableData]
    structured_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# LAYER 3: INTELLIGENCE EXTRACTION LAYER
# ============================================================================

class ExtractedRisk(BaseModel):
    """Extracted risk from filing"""
    risk_type: RiskType
    severity: RiskSeverity
    description: str = Field(..., description="Risk description")
    related_chunks: List[str] = Field(..., description="Chunk IDs supporting this risk")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    impact_area: Optional[str] = None
    mitigation: Optional[str] = None


class FinancialMetric(BaseModel):
    """Extracted financial metric"""
    metric_name: str
    value: float
    unit: str = Field(default="USD", description="Currency or unit")
    period: str = Field(description="Fiscal period")
    change_vs_prior: Optional[float] = Field(None, description="YoY or QoQ change")
    source_chunks: List[str]


class ExtractedEntity(BaseModel):
    """Named entity extracted from filing"""
    entity_type: str = Field(description="executive, subsidiary, partner, competitor")
    name: str
    role: Optional[str] = None
    description: Optional[str] = None
    related_chunks: List[str]


class ExtractedEvent(BaseModel):
    """Business event extracted from filing"""
    event_type: str = Field(description="acquisition, divestiture, partnership, lawsuit, etc")
    description: str
    date: Optional[datetime] = None
    amount: Optional[float] = None
    counterparty: Optional[str] = None
    related_chunks: List[str]


class ExtractedIntelligence(BaseModel):
    """All intelligence extracted from filing"""
    filing_id: str
    metadata: FilingMetadata
    
    risks: List[ExtractedRisk]
    financial_metrics: List[FinancialMetric]
    entities: List[ExtractedEntity]
    events: List[ExtractedEvent]
    
    # Summary
    executive_summary: str = Field(..., description="High-level summary")
    key_highlights: List[str]
    
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# LAYER 4: RETRIEVAL LAYER
# ============================================================================

class RetrievalContext(BaseModel):
    """Context for retrieval"""
    query: str
    ticker: Optional[str] = None
    filing_types: Optional[List[FilingType]] = None
    year_range: Optional[tuple[int, int]] = None
    sections: Optional[List[str]] = None
    risk_types: Optional[List[RiskType]] = None


class RetrievedChunk(BaseModel):
    """Chunk retrieved with score"""
    chunk: StructuredChunk
    relevance_score: float = Field(ge=0, le=1)
    retrieval_method: str = Field(description="bm25, vector, or hybrid")
    reranked: bool = False
    rerank_score: Optional[float] = None


class RetrievalResult(BaseModel):
    """Result of retrieval operation"""
    query: str
    context: RetrievalContext
    chunks: List[RetrievedChunk]
    total_retrieved: int
    retrieval_time_ms: float


# ============================================================================
# LAYER 5: AGENTIC REASONING LAYER
# ============================================================================

class AgentTask(BaseModel):
    """Task for an agent to execute"""
    task_id: str
    agent_type: str = Field(description="filing_intake, structure, risk_extraction, etc")
    task_name: str
    input_data: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentResult(BaseModel):
    """Result from agent execution"""
    task_id: str
    agent_type: str
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class QuestionAnswerRequest(BaseModel):
    """Request for question answering"""
    question: str
    ticker: Optional[str] = None
    filing_types: Optional[List[FilingType]] = None
    year_range: Optional[tuple[int, int]] = None
    use_structured_data: bool = True


class QuestionAnswerResponse(BaseModel):
    """Response to question"""
    question: str
    answer: str
    confidence: float = Field(ge=0, le=1)
    supporting_chunks: List[RetrievedChunk]
    retrieved_filings: List[str]
    data_sources: List[str] = Field(default_factory=list)
    response_time_ms: float = 0.0


class ComparisonRequest(BaseModel):
    """Request for filing comparison"""
    ticker: str
    filing_type: FilingType
    years: List[int] = Field(min_length=2, max_length=5)
    focus_areas: Optional[List[str]] = None


class ComparisonResult(BaseModel):
    """Result of comparison"""
    ticker: str
    years: List[int]
    differences: Dict[str, Any]
    risk_changes: Dict[str, List[ExtractedRisk]]
    metric_trends: Dict[str, List[FinancialMetric]]
    key_changes: List[str]
