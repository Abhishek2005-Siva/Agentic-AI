"""
Layer 1: Ingestion Layer
Fetches, cleans, and stores raw SEC filings
"""
import logging
import re
from datetime import datetime
from typing import Optional, Tuple
import requests
from bs4 import BeautifulSoup

from sec_intelligence.config import get_config
from sec_intelligence.schemas import FilingMetadata, RawFiling, FilingType
from sec_intelligence.storage import get_object_storage, get_postgres_db
from sec_intelligence.storage.models import FilingRecord
from sec_intelligence.utils.helpers import hash_content, generate_id, extract_cik, parse_date

logger = logging.getLogger(__name__)


class FilingFetcher:
    """Fetches SEC filings from EDGAR"""
    
    def __init__(self):
        self.config = get_config()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (SEC Intelligence Platform)"
        })
    
    def fetch_filing(
        self,
        cik: str,
        accession_number: str,
        filing_type: FilingType
    ) -> Optional[Tuple[str, str]]:
        """
        Fetch filing from SEC EDGAR
        Returns: (raw_text, raw_html) or None if failed
        """
        try:
            # Construct URL
            url = f"{self.config.sec.sec_filings_api}/{accession_number.replace('-', '')}/{filing_type.value}-index.html"
            
            logger.info(f"Fetching filing: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract document URL from index
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find main document link
            doc_url = None
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if filing_type.value in href and href.endswith('.htm'):
                    doc_url = f"{self.config.sec.sec_filings_api}/{accession_number.replace('-', '')}/{href}"
                    break
            
            if not doc_url:
                logger.warning(f"Could not find document URL for {accession_number}")
                return None
            
            # Fetch actual document
            doc_response = self.session.get(doc_url, timeout=30)
            doc_response.raise_for_status()
            
            raw_html = doc_response.text
            raw_text = self._extract_text(raw_html)
            
            return raw_text, raw_html
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch filing {accession_number}: {e}")
            return None
    
    def _extract_text(self, html: str) -> str:
        """Extract plain text from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from HTML: {e}")
            return ""
    
    def close(self):
        """Close session"""
        self.session.close()


class MetadataParser:
    """Parses filing metadata"""
    
    @staticmethod
    def extract_from_cik_search(search_result: dict) -> Optional[FilingMetadata]:
        """Extract metadata from CIK search result"""
        try:
            filing_date = search_result.get('filingDate')
            if filing_date:
                filing_date = datetime.strptime(filing_date, '%Y-%m-%d')
            
            # Determine fiscal year from period end
            period_end_str = search_result.get('reportDate', filing_date.isoformat())
            period_end = datetime.strptime(period_end_str, '%Y-%m-%d')
            fiscal_year = period_end.year
            
            metadata = FilingMetadata(
                ticker=search_result.get('ticker', 'UNKNOWN'),
                company_name=search_result.get('companyName', ''),
                filing_type=FilingType(search_result.get('filingType')),
                cik=search_result.get('cikNumber', ''),
                filing_date=filing_date,
                fiscal_year=fiscal_year,
                period_end=period_end,
                accession_number=search_result.get('accessionNumber', ''),
                url=search_result.get('url', '')
            )
            return metadata
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")
            return None
    
    @staticmethod
    def extract_from_html(html: str, ticker: str) -> dict:
        """Extract metadata from filing HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            metadata = {
                'ticker': ticker,
                'company_name': '',
                'cik': '',
                'filing_type': '',
            }
            
            # Try to extract from header
            header = soup.find('div', class_='info')
            if header:
                text = header.get_text()
                lines = text.split('\n')
                for line in lines:
                    if 'CENTRAL INDEX KEY' in line:
                        # Extract CIK
                        parts = line.split(':')
                        if len(parts) > 1:
                            metadata['cik'] = extract_cik(parts[1])
            
            return metadata
        except Exception as e:
            logger.error(f"Failed to extract HTML metadata: {e}")
            return {}


class HTMLCleaner:
    """Cleans HTML from SEC filings"""
    
    @staticmethod
    def clean(html: str) -> str:
        """Clean HTML content"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted tags
            for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
                tag.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, str) and '<!--' in text):
                comment.extract()
            
            return str(soup)
        except Exception as e:
            logger.error(f"Failed to clean HTML: {e}")
            return html


class IngestionAgent:
    """
    Layer 1: Ingestion Agent
    Orchestrates filing acquisition, normalization, and storage
    """
    
    def __init__(self):
        self.fetcher = FilingFetcher()
        self.parser = MetadataParser()
        self.cleaner = HTMLCleaner()
        self.storage = get_object_storage()
        self.db = get_postgres_db()
        self.config = get_config()
    
    def ingest_filing(
        self,
        ticker: str,
        cik: str,
        accession_number: str,
        filing_type: FilingType,
        metadata: Optional[FilingMetadata] = None
    ) -> Optional[RawFiling]:
        """
        Ingest a single filing
        
        Args:
            ticker: Stock ticker
            cik: Central Index Key
            accession_number: SEC accession number
            filing_type: Type of filing
            metadata: Optional pre-parsed metadata
        
        Returns:
            RawFiling object or None if failed
        """
        try:
            logger.info(f"Ingesting filing: {ticker} {accession_number}")
            
            # Fetch from SEC
            fetch_result = self.fetcher.fetch_filing(cik, accession_number, filing_type)
            if not fetch_result:
                logger.error(f"Failed to fetch {accession_number}")
                return None
            
            raw_text, raw_html = fetch_result
            
            # Parse metadata if not provided
            if not metadata:
                metadata_dict = self.parser.extract_from_html(raw_html, ticker)
                # Create basic metadata
                metadata = FilingMetadata(
                    ticker=ticker,
                    company_name=ticker,  # Placeholder
                    filing_type=filing_type,
                    cik=cik,
                    filing_date=datetime.utcnow(),
                    fiscal_year=datetime.utcnow().year,
                    period_end=datetime.utcnow(),
                    accession_number=accession_number,
                    url=""
                )
            
            # Clean HTML
            cleaned_html = self.cleaner.clean(raw_html)
            
            # Calculate hash
            source_hash = hash_content(raw_text)
            
            # Create RawFiling object
            raw_filing = RawFiling(
                metadata=metadata,
                raw_text=raw_text,
                raw_html=cleaned_html,
                source_hash=source_hash
            )
            
            # Store raw content
            self.storage.save_raw(accession_number, raw_html)
            
            # Store in database
            self._save_to_db(raw_filing)
            
            logger.info(f"Successfully ingested: {accession_number}")
            return raw_filing
            
        except Exception as e:
            logger.error(f"Ingestion failed for {accession_number}: {e}")
            return None
    
    def _save_to_db(self, raw_filing: RawFiling):
        """Save filing record to database"""
        try:
            session = self.db.get_session()
            
            filing_record = FilingRecord(
                id=generate_id("filing"),
                ticker=raw_filing.metadata.ticker,
                company_name=raw_filing.metadata.company_name,
                filing_type=raw_filing.metadata.filing_type.value,
                cik=raw_filing.metadata.cik,
                filing_date=raw_filing.metadata.filing_date,
                fiscal_year=raw_filing.metadata.fiscal_year,
                period_end=raw_filing.metadata.period_end,
                accession_number=raw_filing.metadata.accession_number,
                url=raw_filing.metadata.url,
                raw_text_path=str(self.storage.get_raw_path(raw_filing.metadata.accession_number)),
                source_hash=raw_filing.source_hash,
                ingested_at=raw_filing.ingested_at
            )
            
            session.add(filing_record)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            session.rollback()
            session.close()
    
    def close(self):
        """Cleanup"""
        self.fetcher.close()


# Singleton instance
_ingestion_agent: Optional[IngestionAgent] = None


def get_ingestion_agent() -> IngestionAgent:
    """Get ingestion agent instance"""
    global _ingestion_agent
    if _ingestion_agent is None:
        _ingestion_agent = IngestionAgent()
    return _ingestion_agent
