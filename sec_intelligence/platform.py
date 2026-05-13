"""
SEC Intelligence Platform - Main API
Entry point for using the system
"""
import logging
from typing import Optional, List

from sec_intelligence.config import get_config
from sec_intelligence.storage import init_all_storage
from sec_intelligence.schemas import (
    FilingType, QuestionAnswerRequest, QuestionAnswerResponse,
    ComparisonRequest, ComparisonResult
)
from sec_intelligence.core.orchestrator import get_orchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SECIntelligencePlatform:
    """
    Main API for SEC Intelligence Platform
    
    Usage:
        platform = SECIntelligencePlatform()
        platform.initialize()
        
        # Process a filing
        platform.process_filing("NVDA", "0001045810", "0001047469-25-001234", "10-K")
        
        # Answer a question
        response = platform.ask_question("What are the main risks?", ticker="NVDA")
        
        # Compare filings
        comparison = platform.compare_filings("NVDA", ["2024", "2025"])
    """
    
    def __init__(self):
        self.config = get_config()
        self.orchestrator = get_orchestrator()
        self.initialized = False
    
    def initialize(self):
        """Initialize the platform"""
        try:
            logger.info("Initializing SEC Intelligence Platform...")
            
            # Initialize all storage layers
            init_all_storage()
            
            self.initialized = True
            logger.info("Platform initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def process_filing(
        self,
        ticker: str,
        cik: str,
        accession_number: str,
        filing_type: str
    ) -> bool:
        """
        Process a single SEC filing through the complete pipeline
        
        Args:
            ticker: Stock ticker (e.g., "NVDA")
            cik: Central Index Key
            accession_number: SEC accession number
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
        
        Returns:
            True if successful, False otherwise
        
        Example:
            success = platform.process_filing(
                "NVDA",
                "0001045810",
                "0001047469-25-001234",
                "10-K"
            )
        """
        if not self.initialized:
            raise RuntimeError("Platform not initialized. Call initialize() first.")
        
        try:
            filing_enum = FilingType(filing_type)
            return self.orchestrator.process_filing_pipeline(
                ticker, cik, accession_number, filing_enum
            )
        except ValueError:
            logger.error(f"Invalid filing type: {filing_type}")
            return False
    
    def ask_question(
        self,
        question: str,
        ticker: Optional[str] = None,
        filing_types: Optional[List[str]] = None,
        year_range: Optional[tuple] = None
    ) -> QuestionAnswerResponse:
        """
        Ask a question about SEC filings
        
        Args:
            question: The question to ask
            ticker: Optional ticker to focus on
            filing_types: Optional list of filing types to search
            year_range: Optional (start_year, end_year) tuple
        
        Returns:
            QuestionAnswerResponse with answer and supporting evidence
        
        Example:
            response = platform.ask_question(
                "What are the main cybersecurity risks?",
                ticker="NVDA",
                filing_types=["10-K"],
                year_range=(2024, 2025)
            )
            print(response.answer)
        """
        if not self.initialized:
            raise RuntimeError("Platform not initialized. Call initialize() first.")
        
        try:
            filing_enums = None
            if filing_types:
                filing_enums = [FilingType(ft) for ft in filing_types]
            
            request = QuestionAnswerRequest(
                question=question,
                ticker=ticker,
                filing_types=filing_enums,
                use_structured_data=True
            )
            
            return self.orchestrator.answer_question(request)
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return QuestionAnswerResponse(
                question=question,
                answer=f"Error: {str(e)}",
                confidence=0.0,
                supporting_chunks=[],
                retrieved_filings=[]
            )
    
    def compare_filings(
        self,
        ticker: str,
        years: List[int],
        focus_areas: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Compare filings across multiple years
        
        Args:
            ticker: Stock ticker
            years: List of years to compare (e.g., [2023, 2024, 2025])
            focus_areas: Optional focus areas for comparison
        
        Returns:
            ComparisonResult with differences and trends
        
        Example:
            result = platform.compare_filings(
                "NVDA",
                [2023, 2024, 2025],
                focus_areas=["cybersecurity_risks", "revenue", "margins"]
            )
        """
        if not self.initialized:
            raise RuntimeError("Platform not initialized. Call initialize() first.")
        
        try:
            request = ComparisonRequest(
                ticker=ticker,
                filing_type=FilingType.K10,  # Default to 10-K
                years=years,
                focus_areas=focus_areas
            )
            
            return self.orchestrator.compare_filings(request)
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return ComparisonResult(
                ticker=ticker,
                years=years,
                differences={},
                risk_changes={},
                metric_trends={},
                key_changes=[]
            )
    
    def get_company_risks(
        self,
        ticker: str,
        year: Optional[int] = None
    ) -> dict:
        """
        Get structured risk data for a company
        Uses the intelligence extraction layer directly
        
        Args:
            ticker: Stock ticker
            year: Optional specific year (current year if not provided)
        
        Returns:
            Dictionary with risks structured by type and severity
        """
        if not self.initialized:
            raise RuntimeError("Platform not initialized. Call initialize() first.")
        
        # TODO: Implement using structured intelligence queries
        return {
            "ticker": ticker,
            "year": year,
            "risks": []
        }
    
    def get_financial_metrics(
        self,
        ticker: str,
        years: Optional[List[int]] = None
    ) -> dict:
        """
        Get structured financial metrics
        
        Args:
            ticker: Stock ticker
            years: Optional list of years
        
        Returns:
            Dictionary with financial metrics
        """
        if not self.initialized:
            raise RuntimeError("Platform not initialized. Call initialize() first.")
        
        # TODO: Implement using structured metrics queries
        return {
            "ticker": ticker,
            "years": years,
            "metrics": []
        }


def create_platform() -> SECIntelligencePlatform:
    """
    Factory function to create and initialize platform
    
    Example:
        platform = create_platform()
        platform.ask_question("What are NVDA's main risks?", ticker="NVDA")
    """
    platform = SECIntelligencePlatform()
    platform.initialize()
    return platform
