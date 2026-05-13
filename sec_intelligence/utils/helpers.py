"""
Core utilities for SEC Intelligence Platform
"""
import hashlib
import logging
from datetime import datetime
from typing import Optional
import uuid

logger = logging.getLogger(__name__)


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID"""
    uid = str(uuid.uuid4()).replace("-", "")[:12]
    return f"{prefix}_{uid}" if prefix else uid


def hash_content(content: str) -> str:
    """Generate SHA256 hash of content"""
    return hashlib.sha256(content.encode()).hexdigest()


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string from SEC filings"""
    if not date_str:
        return None
    
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None


def extract_cik(cik_str: str) -> str:
    """Extract and normalize CIK"""
    # Remove leading zeros and non-numeric characters
    cik_clean = ''.join(c for c in cik_str if c.isdigit()).lstrip('0')
    return cik_clean or "0"


def clean_text(text: str) -> str:
    """Clean text from filing"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep punctuation
    return text


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


class Logger:
    """Configure logging for SEC Intelligence Platform"""
    
    @staticmethod
    def setup(log_level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('/tmp/sec_intelligence.log')
            ]
        )
