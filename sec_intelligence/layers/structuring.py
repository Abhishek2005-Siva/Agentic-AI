"""
Layer 2: Structuring Layer
Detects sections, performs semantic chunking, extracts tables
"""
import logging
import re
from typing import Optional, List, Dict, Tuple, Any
from bs4 import BeautifulSoup, NavigableString
import pandas as pd

from sec_intelligence.schemas import (
    StructuredChunk, StructuredFiling, TableData, RawFiling, FilingType
)
from sec_intelligence.storage import get_postgres_db, get_object_storage
from sec_intelligence.storage.models import ChunkRecord
from sec_intelligence.utils.helpers import generate_id

logger = logging.getLogger(__name__)


class SectionDetector:
    """Detects SEC filing sections"""
    
    # SEC section patterns
    SEC_SECTIONS = {
        "Item 1": r"^\s*ITEM\s+1[.\s](?!A)",
        "Item 1A": r"^\s*ITEM\s+1A[.\s]",
        "Item 1B": r"^\s*ITEM\s+1B[.\s]",
        "Item 2": r"^\s*ITEM\s+2[.\s]",
        "Item 3": r"^\s*ITEM\s+3[.\s]",
        "Item 4": r"^\s*ITEM\s+4[.\s]",
        "Item 5": r"^\s*ITEM\s+5[.\s]",
        "Item 6": r"^\s*ITEM\s+6[.\s]",
        "Item 7": r"^\s*ITEM\s+7[.\s]",
        "Item 7A": r"^\s*ITEM\s+7A[.\s]",
        "Item 8": r"^\s*ITEM\s+8[.\s]",
        "Item 9": r"^\s*ITEM\s+9[.\s]",
        "Item 9A": r"^\s*ITEM\s+9A[.\s]",
        "Item 9B": r"^\s*ITEM\s+9B[.\s]",
        "Item 10": r"^\s*ITEM\s+10[.\s]",
        "Item 11": r"^\s*ITEM\s+11[.\s]",
        "Item 12": r"^\s*ITEM\s+12[.\s]",
        "Item 13": r"^\s*ITEM\s+13[.\s]",
        "Item 14": r"^\s*ITEM\s+14[.\s]",
        "Item 15": r"^\s*ITEM\s+15[.\s]",
    }
    
    @staticmethod
    def detect_sections(text: str) -> List[Tuple[str, int, int]]:
        """
        Detect sections in filing text
        Returns: List of (section_name, start_pos, end_pos)
        """
        sections = []
        text_upper = text.upper()
        
        for section_name, pattern in SectionDetector.SEC_SECTIONS.items():
            matches = list(re.finditer(pattern, text_upper, re.MULTILINE))
            for i, match in enumerate(matches):
                start = match.start()
                # End is start of next match or end of text
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((section_name, start, end))
        
        # Sort by position
        sections.sort(key=lambda x: x[1])
        
        logger.debug(f"Detected {len(sections)} sections")
        return sections
    
    @staticmethod
    def get_section_text(text: str, start: int, end: int) -> str:
        """Extract section text"""
        return text[start:end]


class SemanticChunker:
    """
    Semantic chunking - preserves paragraph, heading, and table structure
    """
    
    CHUNK_TARGET_SIZE = 1000  # Target characters per chunk
    CHUNK_MAX_SIZE = 2000     # Maximum characters per chunk
    MIN_CHUNK_SIZE = 100      # Minimum characters per chunk
    
    @staticmethod
    def chunk_section(
        section_name: str,
        text: str,
        subsection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a section semantically
        
        Returns: List of chunks with metadata
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_heading = None
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect if this is a heading
            is_heading = SemanticChunker._is_heading(para)
            
            # If adding this would exceed max size and we have content, flush
            if len(current_chunk) + len(para) > SemanticChunker.CHUNK_MAX_SIZE and current_chunk:
                if len(current_chunk) >= SemanticChunker.MIN_CHUNK_SIZE:
                    chunks.append({
                        'text': current_chunk,
                        'heading': chunk_heading,
                        'type': 'text'
                    })
                current_chunk = para
                if is_heading:
                    chunk_heading = para
            else:
                if is_heading:
                    chunk_heading = para
                current_chunk += '\n\n' + para
        
        # Add final chunk
        if len(current_chunk) >= SemanticChunker.MIN_CHUNK_SIZE:
            chunks.append({
                'text': current_chunk,
                'heading': chunk_heading,
                'type': 'text'
            })
        
        return chunks
    
    @staticmethod
    def _is_heading(text: str) -> bool:
        """Detect if text is a heading"""
        # Short text in caps is likely heading
        if len(text) < 100 and text.isupper():
            return True
        # Text starting with number and caps is likely heading
        if re.match(r'^\d+[.\s]', text):
            return True
        return False


class TableExtractor:
    """Extracts tables from filings"""
    
    @staticmethod
    def extract_tables(html: str) -> List[TableData]:
        """Extract tables from HTML"""
        tables = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            table_elements = soup.find_all('table')
            
            for i, table_elem in enumerate(table_elements):
                table_data = TableExtractor._parse_table(table_elem, f"table_{i}")
                if table_data:
                    tables.append(table_data)
            
            logger.debug(f"Extracted {len(tables)} tables")
        except Exception as e:
            logger.error(f"Failed to extract tables: {e}")
        
        return tables
    
    @staticmethod
    def _parse_table(table_elem, table_id: str) -> Optional[TableData]:
        """Parse a single table element"""
        try:
            # Try with pandas
            df = pd.read_html(str(table_elem))[0]
            
            # Get headers
            headers = list(df.columns)
            
            # Get rows as dicts
            rows = df.to_dict('records')
            
            # Detect table type
            table_type = TableExtractor._detect_table_type(' '.join(headers))
            
            return TableData(
                table_type=table_type,
                title=None,
                headers=headers,
                rows=rows
            )
        except Exception as e:
            logger.debug(f"Failed to parse table {table_id}: {e}")
            return None
    
    @staticmethod
    def _detect_table_type(header_text: str) -> str:
        """Detect table type from headers"""
        header_upper = header_text.upper()
        
        if 'BALANCE' in header_upper or 'ASSETS' in header_upper:
            return 'balance_sheet'
        if 'REVENUE' in header_upper or 'SALES' in header_upper:
            return 'revenue'
        if 'SEGMENT' in header_upper:
            return 'segment'
        if 'CASH' in header_upper:
            return 'cash_flow'
        
        return 'financial'


class StructuringAgent:
    """
    Layer 2: Structure Agent
    Orchestrates section detection, chunking, and table extraction
    """
    
    def __init__(self):
        self.section_detector = SectionDetector()
        self.chunker = SemanticChunker()
        self.table_extractor = TableExtractor()
        self.db = get_postgres_db()
        self.storage = get_object_storage()
    
    def structure_filing(self, raw_filing: RawFiling) -> Optional[StructuredFiling]:
        """
        Structure a raw filing into chunks and intelligence
        
        Args:
            raw_filing: RawFiling from ingestion layer
        
        Returns:
            StructuredFiling object
        """
        try:
            logger.info(f"Structuring filing: {raw_filing.metadata.accession_number}")
            
            # Extract tables
            tables = self.table_extractor.extract_tables(raw_filing.raw_html or "")
            
            # Detect sections
            text = raw_filing.raw_text
            sections = self.section_detector.detect_sections(text)
            
            chunks = []
            section_map = {}
            
            # Process each section
            for section_name, start, end in sections:
                section_text = self.section_detector.get_section_text(text, start, end)
                
                # Create chunks for this section
                section_chunks = self.chunker.chunk_section(section_name, section_text)
                
                section_chunk_ids = []
                
                for i, chunk_data in enumerate(section_chunks):
                    chunk_id = generate_id("chunk")
                    section_chunk_ids.append(chunk_id)
                    
                    chunk = StructuredChunk(
                        chunk_id=chunk_id,
                        filing_id=generate_id("filing"),  # Will be set by caller
                        section=section_name,
                        subsection=chunk_data.get('heading'),
                        text=chunk_data['text'],
                        chunk_type=chunk_data['type'],
                        ticker=raw_filing.metadata.ticker,
                        filing_type=raw_filing.metadata.filing_type,
                        filing_date=raw_filing.metadata.filing_date,
                        fiscal_year=raw_filing.metadata.fiscal_year,
                        heading=chunk_data.get('heading'),
                        is_table=False,
                        page_number=None,
                        byte_offset=start + text.find(chunk_data['text'])
                    )
                    
                    chunks.append(chunk)
                
                section_map[section_name] = section_chunk_ids
            
            # Create structured filing
            structured_filing = StructuredFiling(
                filing_id=generate_id("filing"),
                metadata=raw_filing.metadata,
                chunks=chunks,
                section_map=section_map,
                tables=tables
            )
            
            logger.info(f"Structured filing into {len(chunks)} chunks")
            return structured_filing
            
        except Exception as e:
            logger.error(f"Failed to structure filing: {e}")
            return None
    
    def _save_chunks_to_db(self, filing_id: str, chunks: List[StructuredChunk]):
        """Save chunks to database"""
        try:
            session = self.db.get_session()
            
            for chunk in chunks:
                chunk_record = ChunkRecord(
                    id=chunk.chunk_id,
                    filing_id=filing_id,
                    section=chunk.section,
                    subsection=chunk.subsection,
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    ticker=chunk.ticker,
                    filing_type=chunk.filing_type.value,
                    filing_date=chunk.filing_date,
                    fiscal_year=chunk.fiscal_year,
                    heading=chunk.heading,
                    is_table=chunk.is_table,
                    table_data=chunk.table_data.dict() if chunk.table_data else None,
                    page_number=chunk.page_number,
                    byte_offset=chunk.byte_offset
                )
                session.add(chunk_record)
            
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            session.rollback()
            session.close()


# Singleton instance
_structuring_agent: Optional[StructuringAgent] = None


def get_structuring_agent() -> StructuringAgent:
    """Get structuring agent instance"""
    global _structuring_agent
    if _structuring_agent is None:
        _structuring_agent = StructuringAgent()
    return _structuring_agent
