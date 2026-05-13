"""
Layer 3: Intelligence Extraction Layer
Extracts risks, financial metrics, entities, and events
"""
import logging
import re
from typing import Optional, List
from openai import OpenAI

from sec_intelligence.schemas import (
    StructuredFiling, ExtractedIntelligence, ExtractedRisk, FinancialMetric,
    ExtractedEntity, ExtractedEvent, RiskSeverity, RiskType
)
from sec_intelligence.config import get_config
from sec_intelligence.storage import get_postgres_db
from sec_intelligence.storage.models import IntelligenceRecord, RiskRecord, MetricRecord, EntityRecord, EventRecord
from sec_intelligence.utils.helpers import generate_id

logger = logging.getLogger(__name__)


class RiskExtractionAgent:
    """Extracts risks from filings"""
    
    def __init__(self):
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.llm.api_key)
        
        self.risk_focus_areas = {
            RiskType.CYBERSECURITY: ["cybersecurity", "data breach", "ransomware", "hacking", "security"],
            RiskType.SUPPLY_CHAIN: ["supply chain", "supplier", "logistics", "disruption"],
            RiskType.GEOPOLITICAL: ["geopolitical", "trade", "sanction", "tariff", "war"],
            RiskType.AI_REGULATION: ["AI", "artificial intelligence", "regulation", "compliance"],
            RiskType.LIQUIDITY: ["liquidity", "cash flow", "debt", "financing"],
            RiskType.LITIGATION: ["litigation", "lawsuit", "legal", "dispute"],
            RiskType.OPERATIONAL: ["operational", "supply", "production", "capacity"],
        }
    
    def extract_risks(self, structured_filing: StructuredFiling) -> List[ExtractedRisk]:
        """
        Extract risks from filing
        Uses GPT-4o mini to identify and categorize risks
        """
        risks = []
        
        try:
            # Get risk-related chunks (usually in Item 1A)
            risk_chunks = [
                chunk for chunk in structured_filing.chunks
                if "Item 1A" in chunk.section or "risk" in chunk.text.lower()
            ]
            
            if not risk_chunks:
                logger.warning("No risk-related chunks found")
                return risks
            
            # Prepare text for analysis
            risk_text = "\n\n".join([chunk.text for chunk in risk_chunks[:5]])  # Limit to first 5
            
            # Call GPT-4o mini to extract risks
            response = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": self._build_risk_extraction_prompt(risk_text, structured_filing.metadata.ticker)
                }]
            )
            
            # Parse response
            extracted = self._parse_risk_response(response.choices[0].message.content, risk_chunks)
            risks.extend(extracted)
            
            logger.info(f"Extracted {len(risks)} risks")
            return risks
            
        except Exception as e:
            logger.error(f"Risk extraction failed: {e}")
            return risks
    
    def _build_risk_extraction_prompt(self, text: str, ticker: str) -> str:
        """Build prompt for risk extraction"""
        return f"""Analyze the following SEC filing excerpt for {ticker} and extract key risks.

For each risk identified, provide:
1. Risk Type (one of: {', '.join([t.value for t in RiskType])})
2. Severity (low, medium, high, critical)
3. Description (2-3 sentences)
4. Impact Area

Format as JSON array:
[
  {{"type": "...", "severity": "...", "description": "...", "impact_area": "..."}},
  ...
]

FILING TEXT:
{text}

Return ONLY valid JSON, no other text."""

    def _parse_risk_response(self, response_text: str, chunk_refs: List) -> List[ExtractedRisk]:
        """Parse GPT response into risks"""
        risks = []
        
        try:
            # Extract JSON from response
            import json
            # Try to find JSON array
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not match:
                return risks
            
            json_str = match.group(0)
            data = json.loads(json_str)
            
            for item in data:
                try:
                    risk = ExtractedRisk(
                        risk_type=RiskType(item['type'].lower().replace(' ', '_')),
                        severity=RiskSeverity(item['severity'].lower()),
                        description=item['description'],
                        related_chunks=[c.chunk_id for c in chunk_refs[:3]],
                        confidence=0.8,
                        impact_area=item.get('impact_area'),
                        mitigation=item.get('mitigation')
                    )
                    risks.append(risk)
                except (KeyError, ValueError) as e:
                    logger.debug(f"Failed to parse risk item: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to parse risk response: {e}")
        
        return risks


class FinancialMetricsAgent:
    """Extracts financial metrics from filings"""
    
    def __init__(self):
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.llm.api_key)
        
        self.key_metrics = [
            "revenue", "gross_profit", "operating_income", "net_income",
            "total_assets", "total_liabilities", "stockholders_equity",
            "operating_margin", "net_margin", "return_on_assets",
            "earnings_per_share", "cash_flow"
        ]
    
    def extract_metrics(self, structured_filing: StructuredFiling) -> List[FinancialMetric]:
        """Extract financial metrics"""
        metrics = []
        
        try:
            # Get financial statement chunks (usually Items 6, 7, 8)
            financial_chunks = [
                chunk for chunk in structured_filing.chunks
                if any(item in chunk.section for item in ["Item 6", "Item 7", "Item 8"])
            ]
            
            if not financial_chunks:
                logger.warning("No financial chunks found")
                return metrics
            
            # Extract from tables
            for table in structured_filing.tables:
                if 'financial' in table.table_type or table.table_type == 'balance_sheet':
                    extracted = self._extract_from_table(table, structured_filing.metadata.fiscal_year)
                    metrics.extend(extracted)
            
            logger.info(f"Extracted {len(metrics)} financial metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Financial metrics extraction failed: {e}")
            return metrics
    
    def _extract_from_table(self, table, fiscal_year: int) -> List[FinancialMetric]:
        """Extract metrics from a table"""
        metrics = []
        
        try:
            for row in table.rows:
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        metric = FinancialMetric(
                            metric_name=key,
                            value=float(value),
                            unit="USD",
                            period=str(fiscal_year),
                            source_chunks=[]
                        )
                        metrics.append(metric)
        except Exception as e:
            logger.debug(f"Failed to extract from table: {e}")
        
        return metrics


class EntityExtractionAgent:
    """Extracts named entities from filings"""
    
    def __init__(self):
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.llm.api_key)
    
    def extract_entities(self, structured_filing: StructuredFiling) -> List[ExtractedEntity]:
        """Extract entities like executives, subsidiaries"""
        entities = []
        
        try:
            # Get relevant chunks for entities (Item 1, Item 10)
            entity_chunks = [
                chunk for chunk in structured_filing.chunks
                if any(item in chunk.section for item in ["Item 1", "Item 10"])
            ]
            
            if not entity_chunks:
                return entities
            
            # Use GPT-4o mini to extract
            text = "\n\n".join([chunk.text for chunk in entity_chunks[:3]])
            
            response = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": self._build_entity_extraction_prompt(text)
                }]
            )
            
            extracted = self._parse_entity_response(response.choices[0].message.content, entity_chunks)
            entities.extend(extracted)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return entities
    
    def _build_entity_extraction_prompt(self, text: str) -> str:
        """Build entity extraction prompt"""
        return f"""Extract named entities from this SEC filing excerpt.

Extract:
- Executives (CEO, CFO, etc.) with their roles
- Subsidiaries
- Major partners
- Competitors mentioned

Format as JSON:
{{
  "executives": [{{"name": "...", "role": "..."}}],
  "subsidiaries": [{{"name": "...", "description": "..."}}],
  "partners": [{{"name": "...", "type": "..."}}],
  "competitors": [...]
}}

FILING TEXT:
{text}

Return ONLY valid JSON."""

    def _parse_entity_response(self, response_text: str, chunk_refs: List) -> List[ExtractedEntity]:
        """Parse entity extraction response"""
        entities = []
        
        try:
            import json
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                return entities
            
            data = json.loads(match.group(0))
            
            # Process executives
            for exec_data in data.get('executives', []):
                entity = ExtractedEntity(
                    entity_type='executive',
                    name=exec_data.get('name'),
                    role=exec_data.get('role'),
                    related_chunks=[c.chunk_id for c in chunk_refs[:1]]
                )
                entities.append(entity)
            
            # Process subsidiaries
            for sub_data in data.get('subsidiaries', []):
                entity = ExtractedEntity(
                    entity_type='subsidiary',
                    name=sub_data.get('name'),
                    description=sub_data.get('description'),
                    related_chunks=[c.chunk_id for c in chunk_refs[:1]]
                )
                entities.append(entity)
        
        except Exception as e:
            logger.debug(f"Failed to parse entity response: {e}")
        
        return entities


class EventExtractionAgent:
    """Extracts business events from filings"""
    
    def __init__(self):
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.llm.api_key)
    
    def extract_events(self, structured_filing: StructuredFiling) -> List[ExtractedEvent]:
        """Extract business events"""
        events = []
        
        try:
            # Get relevant chunks
            event_chunks = [
                chunk for chunk in structured_filing.chunks
                if any(word in chunk.text.lower() for word in 
                      ["acquisition", "merger", "divestiture", "lawsuit", "settlement", "partnership"])
            ]
            
            if not event_chunks:
                return events
            
            # Use GPT-4o mini
            text = "\n\n".join([chunk.text for chunk in event_chunks[:5]])
            
            response = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": self._build_event_extraction_prompt(text)
                }]
            )
            
            extracted = self._parse_event_response(response.choices[0].message.content, event_chunks)
            events.extend(extracted)
            
            return events
            
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return events
    
    def _build_event_extraction_prompt(self, text: str) -> str:
        """Build event extraction prompt"""
        return f"""Extract business events from this SEC filing excerpt.

Extract significant events like:
- Acquisitions/mergers
- Divestitures
- Partnerships
- Lawsuits/settlements
- Product launches
- Facility closures

Format as JSON array:
[
  {{"event_type": "...", "description": "...", "date": "...", "counterparty": "...", "amount": ...}},
  ...
]

FILING TEXT:
{text}

Return ONLY valid JSON."""

    def _parse_event_response(self, response_text: str, chunk_refs: List) -> List[ExtractedEvent]:
        """Parse event extraction response"""
        events = []
        
        try:
            import json
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not match:
                return events
            
            data = json.loads(match.group(0))
            
            for event_data in data:
                event = ExtractedEvent(
                    event_type=event_data.get('event_type', 'unknown'),
                    description=event_data.get('description', ''),
                    date=None,  # Could parse date if provided
                    amount=event_data.get('amount'),
                    counterparty=event_data.get('counterparty'),
                    related_chunks=[c.chunk_id for c in chunk_refs[:2]]
                )
                events.append(event)
        
        except Exception as e:
            logger.debug(f"Failed to parse event response: {e}")
        
        return events


class IntelligenceExtractionAgent:
    """
    Layer 3: Intelligence Extraction Agent
    Orchestrates all extraction agents and validates results
    """
    
    def __init__(self):
        self.config = get_config()
        self.client = OpenAI(api_key=self.config.llm.api_key)
        self.risk_agent = RiskExtractionAgent()
        self.financial_agent = FinancialMetricsAgent()
        self.entity_agent = EntityExtractionAgent()
        self.event_agent = EventExtractionAgent()
        self.db = get_postgres_db()
    
    def extract_intelligence(
        self,
        structured_filing: StructuredFiling
    ) -> Optional[ExtractedIntelligence]:
        """
        Extract all intelligence from structured filing
        
        Args:
            structured_filing: StructuredFiling from Layer 2
        
        Returns:
            ExtractedIntelligence object
        """
        try:
            logger.info(f"Extracting intelligence: {structured_filing.metadata.accession_number}")
            
            # Extract in parallel conceptually (sequential for now)
            risks = self.risk_agent.extract_risks(structured_filing)
            metrics = self.financial_agent.extract_metrics(structured_filing)
            entities = self.entity_agent.extract_entities(structured_filing)
            events = self.event_agent.extract_events(structured_filing)
            
            # Generate executive summary
            summary = self._generate_summary(structured_filing, risks, metrics)
            
            # Extract key highlights
            highlights = self._extract_highlights(risks, metrics, events)
            
            # Create intelligence object
            intelligence = ExtractedIntelligence(
                filing_id=structured_filing.filing_id,
                metadata=structured_filing.metadata,
                risks=risks,
                financial_metrics=metrics,
                entities=entities,
                events=events,
                executive_summary=summary,
                key_highlights=highlights
            )
            
            logger.info(f"Extracted intelligence: {len(risks)} risks, {len(metrics)} metrics, {len(entities)} entities")
            return intelligence
            
        except Exception as e:
            logger.error(f"Intelligence extraction failed: {e}")
            return None
    
    def _generate_summary(self, filing: StructuredFiling, risks: List[ExtractedRisk], metrics: List[FinancialMetric]) -> str:
        """Generate executive summary"""
        try:
            risk_text = f"Key risks identified: {', '.join([r.risk_type.value for r in risks[:3]])}" if risks else "No major risks identified"
            
            response = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""Write a 2-3 sentence executive summary for {filing.metadata.ticker}'s
                    {filing.metadata.filing_type.value} filing. {risk_text}.
                    Key focus areas: financial performance, major risks, strategic initiatives."""
                }]
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Filing analysis not available"
    
    def _extract_highlights(self, risks: List[ExtractedRisk], metrics: List[FinancialMetric], events: List[ExtractedEvent]) -> List[str]:
        """Extract key highlights"""
        highlights = []
        
        # Add top risks
        critical_risks = [r for r in risks if r.severity.value == "critical"]
        for risk in critical_risks[:2]:
            highlights.append(f"Critical risk: {risk.description}")
        
        # Add top metrics
        if metrics:
            highlights.append(f"Reported {len(metrics)} key financial metrics")
        
        # Add events
        for event in events[:2]:
            highlights.append(f"{event.event_type.upper()}: {event.description}")
        
        return highlights


# Singleton instance
_intelligence_agent: Optional[IntelligenceExtractionAgent] = None


def get_intelligence_extraction_agent() -> IntelligenceExtractionAgent:
    """Get intelligence extraction agent instance"""
    global _intelligence_agent
    if _intelligence_agent is None:
        _intelligence_agent = IntelligenceExtractionAgent()
    return _intelligence_agent
