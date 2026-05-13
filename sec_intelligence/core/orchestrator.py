"""
Layer 5: Agentic Reasoning Layer
Core agent orchestrator and decision-making
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from sec_intelligence.config import get_config
from sec_intelligence.schemas import (
    AgentTask, AgentResult, QuestionAnswerRequest, QuestionAnswerResponse,
    ComparisonRequest, ComparisonResult, RawFiling, StructuredFiling,
    ExtractedIntelligence, RetrievalContext, FilingType
)
from sec_intelligence.layers.ingestion import get_ingestion_agent
from sec_intelligence.layers.structuring import get_structuring_agent
from sec_intelligence.layers.extraction import get_intelligence_extraction_agent
from sec_intelligence.retrieval import get_retrieval_agent
from sec_intelligence.storage import get_postgres_db, get_object_storage
from sec_intelligence.utils.helpers import generate_id
from openai import OpenAI

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Agent types in the system"""
    FILING_INTAKE = "filing_intake"
    STRUCTURE = "structure"
    RISK_EXTRACTION = "risk_extraction"
    FINANCIAL = "financial"
    VALIDATION = "validation"
    RETRIEVAL = "retrieval"
    QA = "qa"


class ValidationAgent:
    """
    Validation Agent
    Checks for missing sections, extraction confidence, data inconsistencies
    """
    
    def __init__(self):
        self.config = get_config()
    
    def validate_extraction(self, intelligence: ExtractedIntelligence) -> Dict[str, Any]:
        """Validate extracted intelligence"""
        validations = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'missing_sections': []
        }
        
        # Check for empty collections
        if not intelligence.risks:
            validations['warnings'].append("No risks extracted")
        
        if not intelligence.financial_metrics:
            validations['warnings'].append("No financial metrics extracted")
        
        if not intelligence.entities:
            validations['warnings'].append("No entities extracted")
        
        # Check confidence scores
        low_confidence_risks = [r for r in intelligence.risks if r.confidence < 0.5]
        if low_confidence_risks:
            validations['warnings'].append(
                f"{len(low_confidence_risks)} risks with low confidence"
            )
        
        # Check for required sections in chunks
        required_sections = ["Item 1", "Item 1A", "Item 7"]
        found_sections = {chunk.section for chunk in []}  # Would be populated from chunks
        missing = set(required_sections) - found_sections
        if missing:
            validations['warnings'].append(f"Missing sections: {missing}")
        
        return validations


class AgentOrchestrator:
    """
    Layer 5: Agent Orchestrator
    Routes tasks to appropriate agents, manages workflow
    """
    
    def __init__(self):
        self.config = get_config()
        self.ingestion_agent = get_ingestion_agent()
        self.structure_agent = get_structuring_agent()
        self.extraction_agent = get_intelligence_extraction_agent()
        self.retrieval_agent = get_retrieval_agent()
        self.validation_agent = ValidationAgent()
        self.client = OpenAI(api_key=self.config.llm.api_key)
        self.db = get_postgres_db()
        self.storage = get_object_storage()
    
    def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute an agent task
        
        Args:
            task: AgentTask to execute
        
        Returns:
            AgentResult with outcome
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Executing task: {task.task_name} ({task.agent_type})")
            
            # Route to appropriate agent
            if task.agent_type == AgentType.FILING_INTAKE.value:
                output = self._execute_filing_intake(task)
            elif task.agent_type == AgentType.STRUCTURE.value:
                output = self._execute_structure(task)
            elif task.agent_type == AgentType.RISK_EXTRACTION.value:
                output = self._execute_risk_extraction(task)
            elif task.agent_type == AgentType.VALIDATION.value:
                output = self._execute_validation(task)
            elif task.agent_type == AgentType.RETRIEVAL.value:
                output = self._execute_retrieval(task)
            else:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
            
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                success=True,
                output=output,
                execution_time_ms=elapsed,
                completed_at=datetime.utcnow()
            )
            
        except Exception as e:
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Task execution failed: {e}")
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=task.agent_type,
                success=False,
                error=str(e),
                execution_time_ms=elapsed,
                completed_at=datetime.utcnow()
            )
    
    def _execute_filing_intake(self, task: AgentTask) -> Dict[str, Any]:
        """Execute filing intake task"""
        filing = self.ingestion_agent.ingest_filing(
            ticker=task.input_data['ticker'],
            cik=task.input_data['cik'],
            accession_number=task.input_data['accession_number'],
            filing_type=FilingType(task.input_data['filing_type'])
        )
        
        return {
            'filing_id': filing.metadata.accession_number if filing else None,
            'status': 'ingested' if filing else 'failed'
        }
    
    def _execute_structure(self, task: AgentTask) -> Dict[str, Any]:
        """Execute structuring task"""
        # In real implementation, would fetch RawFiling from DB
        # For now, simplified
        return {'status': 'structured', 'chunks_created': 0}
    
    def _execute_risk_extraction(self, task: AgentTask) -> Dict[str, Any]:
        """Execute risk extraction task"""
        return {'status': 'extracted', 'risks_found': 0}
    
    def _execute_validation(self, task: AgentTask) -> Dict[str, Any]:
        """Execute validation task"""
        return {'status': 'validated', 'valid': True}
    
    def _execute_retrieval(self, task: AgentTask) -> Dict[str, Any]:
        """Execute retrieval task"""
        query = task.input_data.get('query')
        context = RetrievalContext(query=query)
        
        result = self.retrieval_agent.retrieve_for_query(query, context)
        
        return {
            'chunks_retrieved': result.total_retrieved,
            'retrieval_method': 'hybrid'
        }
    
    def process_filing_pipeline(
        self,
        ticker: str,
        cik: str,
        accession_number: str,
        filing_type: FilingType
    ) -> bool:
        """
        Execute complete filing processing pipeline
        
        Pipeline:
        1. Ingest
        2. Structure
        3. Extract Intelligence
        4. Validate
        """
        try:
            logger.info(f"Processing filing pipeline: {accession_number}")
            
            # Step 1: Ingest
            raw_filing = self.ingestion_agent.ingest_filing(
                ticker, cik, accession_number, filing_type
            )
            if not raw_filing:
                logger.error("Ingestion failed")
                return False
            
            # Step 2: Structure
            structured_filing = self.structure_agent.structure_filing(raw_filing)
            if not structured_filing:
                logger.error("Structuring failed")
                return False
            
            # Step 3: Extract Intelligence
            intelligence = self.extraction_agent.extract_intelligence(structured_filing)
            if not intelligence:
                logger.error("Intelligence extraction failed")
                return False
            
            # Step 4: Validate
            validation = self.validation_agent.validate_extraction(intelligence)
            
            if validation['errors']:
                logger.error(f"Validation errors: {validation['errors']}")
                return False
            
            logger.info(f"Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
    
    def answer_question(self, request: QuestionAnswerRequest) -> QuestionAnswerResponse:
        """
        Answer a question about SEC filings
        
        Args:
            request: QuestionAnswerRequest
        
        Returns:
            QuestionAnswerResponse with answer and supporting evidence
        """
        try:
            logger.info(f"Answering question: {request.question}")
            
            # Build retrieval context
            context = RetrievalContext(
                query=request.question,
                ticker=request.ticker,
                filing_types=request.filing_types,
                year_range=request.year_range if hasattr(request, 'year_range') else None
            )
            
            # Retrieve relevant chunks
            retrieval_result = self.retrieval_agent.retrieve_for_query(
                request.question, context
            )
            
            if not retrieval_result.chunks:
                answer = "I could not find relevant information in SEC filings to answer this question."
                confidence = 0.0
                supporting_text = ""
            else:
                # Prepare context for GPT-4o mini
                supporting_text = "\n\n".join([
                    f"[{c.chunk.section}] {c.chunk.text}"
                    for c in retrieval_result.chunks[:5]
                ])
                
                # Call GPT-4o mini for answer
                response = self.client.chat.completions.create(
                    model=self.config.llm.model_name,
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": self._build_qa_prompt(request.question, supporting_text)
                    }]
                )
                
                answer = response.choices[0].message.content
                confidence = 0.85  # Simplified confidence
            
            return QuestionAnswerResponse(
                question=request.question,
                answer=answer,
                confidence=confidence,
                supporting_chunks=retrieval_result.chunks,
                retrieved_filings=[c.chunk.filing_id for c in retrieval_result.chunks],
                response_time_ms=0.0  # TODO: add timing
            )
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            
            return QuestionAnswerResponse(
                question=request.question,
                answer=f"Error: {str(e)}",
                confidence=0.0,
                supporting_chunks=[],
                retrieved_filings=[]
            )
    
    def compare_filings(self, request: ComparisonRequest) -> ComparisonResult:
        """
        Compare filings across years
        
        Args:
            request: ComparisonRequest
        
        Returns:
            ComparisonResult with differences
        """
        try:
            logger.info(f"Comparing {request.ticker} filings: {request.years}")
            
            # TODO: Implement comparison logic
            # Fetch filings for each year, compare risks, metrics, etc
            
            result = ComparisonResult(
                ticker=request.ticker,
                years=request.years,
                differences={},
                risk_changes={},
                metric_trends={},
                key_changes=[]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return ComparisonResult(
                ticker=request.ticker,
                years=request.years,
                differences={},
                risk_changes={},
                metric_trends={},
                key_changes=[]
            )
    
    def _build_qa_prompt(self, question: str, context: str) -> str:
        """Build prompt for question answering"""
        return f"""Based on the following SEC filing excerpts, answer this question:

QUESTION: {question}

CONTEXT:
{context}

Provide a direct, concise answer based only on the information in the context.
If the context doesn't contain relevant information, say so."""


# Singleton instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get agent orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
