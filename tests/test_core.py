"""
Unit tests for SEC Intelligence Platform
"""

import pytest
from datetime import datetime
from sec_intelligence.schemas import (
    FilingMetadata, RawFiling, StructuredChunk, FilingType,
    ExtractedRisk, RiskSeverity, RiskType, FinancialMetric
)
from sec_intelligence.utils.helpers import (
    generate_id, hash_content, clean_text, extract_cik
)


class TestUtilities:
    """Test utility functions"""
    
    def test_generate_id(self):
        """Test ID generation"""
        id1 = generate_id("filing")
        id2 = generate_id("filing")
        
        assert id1.startswith("filing_")
        assert id2.startswith("filing_")
        assert id1 != id2
        assert len(id1) > 10
    
    def test_hash_content(self):
        """Test content hashing"""
        content = "This is test content"
        hash1 = hash_content(content)
        hash2 = hash_content(content)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256
        
        # Different content produces different hash
        assert hash_content("Different content") != hash1
    
    def test_clean_text(self):
        """Test text cleaning"""
        messy = "  This   is   messy    text  with  spaces  "
        clean = clean_text(messy)
        
        assert "   " not in clean
        assert clean == "This is messy text with spaces"
    
    def test_extract_cik(self):
        """Test CIK extraction"""
        assert extract_cik("0001045810") == "1045810"
        assert extract_cik("1045810") == "1045810"
        assert extract_cik("00000001") == "1"


class TestSchemas:
    """Test data schemas"""
    
    def test_filing_metadata(self):
        """Test FilingMetadata schema"""
        metadata = FilingMetadata(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            filing_type=FilingType.K10,
            cik="0001045810",
            filing_date=datetime(2025, 2, 27),
            fiscal_year=2025,
            period_end=datetime(2025, 1, 26),
            accession_number="0001047469-25-001234",
            url="https://www.sec.gov/Archives/..."
        )
        
        assert metadata.ticker == "NVDA"
        assert metadata.filing_type == FilingType.K10
        assert metadata.fiscal_year == 2025
    
    def test_raw_filing(self):
        """Test RawFiling schema"""
        metadata = FilingMetadata(
            ticker="NVDA",
            company_name="NVIDIA",
            filing_type=FilingType.K10,
            cik="0001045810",
            filing_date=datetime(2025, 2, 27),
            fiscal_year=2025,
            period_end=datetime(2025, 1, 26),
            accession_number="0001047469-25-001234",
            url=""
        )
        
        raw_filing = RawFiling(
            metadata=metadata,
            raw_text="Test filing text",
            raw_html="<html>Test filing</html>",
            source_hash="abc123"
        )
        
        assert raw_filing.metadata.ticker == "NVDA"
        assert raw_filing.raw_text == "Test filing text"
    
    def test_structured_chunk(self):
        """Test StructuredChunk schema"""
        chunk = StructuredChunk(
            chunk_id="chunk_abc123",
            filing_id="filing_xyz789",
            section="Item 1A",
            subsection="Risk Factors",
            text="Risk factor text here",
            ticker="NVDA",
            filing_type=FilingType.K10,
            filing_date=datetime(2025, 2, 27),
            fiscal_year=2025
        )
        
        assert chunk.chunk_id == "chunk_abc123"
        assert chunk.section == "Item 1A"
        assert chunk.chunk_type == "text"  # Default
    
    def test_extracted_risk(self):
        """Test ExtractedRisk schema"""
        risk = ExtractedRisk(
            risk_type=RiskType.CYBERSECURITY,
            severity=RiskSeverity.HIGH,
            description="Company faces significant cybersecurity threats",
            related_chunks=["chunk_1", "chunk_2"],
            confidence=0.85
        )
        
        assert risk.risk_type == RiskType.CYBERSECURITY
        assert risk.severity == RiskSeverity.HIGH
        assert risk.confidence == 0.85
    
    def test_financial_metric(self):
        """Test FinancialMetric schema"""
        metric = FinancialMetric(
            metric_name="revenue",
            value=60000000000,
            unit="USD",
            period="2025",
            source_chunks=["chunk_1"]
        )
        
        assert metric.metric_name == "revenue"
        assert metric.value == 60000000000


class TestSectionDetector:
    """Test section detection"""
    
    def test_detect_sections(self):
        """Test section detection in text"""
        from sec_intelligence.layers.structuring import SectionDetector
        
        text = """
        ITEM 1 - BUSINESS
        Company description here.
        
        ITEM 1A - RISK FACTORS
        Risk description here.
        
        ITEM 7 - FINANCIAL DATA
        Financial information here.
        """
        
        sections = SectionDetector.detect_sections(text)
        
        assert len(sections) >= 1  # At least one section found
        assert any("Item 1" in s[0] for s in sections)


class TestChunking:
    """Test semantic chunking"""
    
    def test_chunk_section(self):
        """Test chunking logic"""
        from sec_intelligence.layers.structuring import SemanticChunker
        
        text = """
        HEADING ONE
        This is a paragraph about something important.
        
        HEADING TWO
        This is another paragraph with different content.
        """
        
        chunks = SemanticChunker.chunk_section("Item 1", text)
        
        assert len(chunks) > 0
        assert 'text' in chunks[0]
        assert 'type' in chunks[0]


class TestIntegration:
    """Integration tests"""
    
    def test_schema_serialization(self):
        """Test that schemas can be serialized"""
        metadata = FilingMetadata(
            ticker="NVDA",
            company_name="NVIDIA",
            filing_type=FilingType.K10,
            cik="0001045810",
            filing_date=datetime(2025, 2, 27),
            fiscal_year=2025,
            period_end=datetime(2025, 1, 26),
            accession_number="0001047469-25-001234",
            url=""
        )
        
        # Should be serializable to dict
        data = metadata.dict()
        assert data['ticker'] == "NVDA"
        assert data['filing_type'] == "10-K"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
