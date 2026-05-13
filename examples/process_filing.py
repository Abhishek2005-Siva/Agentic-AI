#!/usr/bin/env python
"""
Filing processing example: Process SEC filing through complete pipeline
Run: python examples/process_filing.py
"""

from sec_intelligence import create_platform
from sec_intelligence.schemas import FilingType


def main():
    print("\n" + "=" * 70)
    print(" SEC Intelligence Platform - Filing Processing Pipeline")
    print("=" * 70 + "\n")
    
    # Initialize
    print("Initializing platform...")
    try:
        platform = create_platform()
        print("✓ Platform ready\n")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return
    
    # Filing to process
    filing = {
        "ticker": "NVDA",
        "cik": "0001045810",
        "accession_number": "0001047469-25-002000",  # Example - replace with real
        "filing_type": "10-K"
    }
    
    print("Filing to process:")
    print(f"  Ticker: {filing['ticker']}")
    print(f"  CIK: {filing['cik']}")
    print(f"  Accession: {filing['accession_number']}")
    print(f"  Type: {filing['filing_type']}")
    print()
    
    # Process
    print("Starting pipeline...")
    print("-" * 70)
    
    try:
        success = platform.process_filing(
            ticker=filing['ticker'],
            cik=filing['cik'],
            accession_number=filing['accession_number'],
            filing_type=filing['filing_type']
        )
        
        print("-" * 70)
        
        if success:
            print("\n✓ Pipeline completed successfully!")
            print("\nFiling has been processed through:")
            print("  1. Ingestion Layer     - Filing fetched and cleaned")
            print("  2. Structuring Layer   - Sections detected, chunks created")
            print("  3. Extraction Layer    - Risks, metrics, entities extracted")
            print("  4. Storage Layer       - Indexed in PostgreSQL + Qdrant")
            print("\nYou can now ask questions about this filing:")
            print(f"  Q: What are {filing['ticker']}'s main risks?")
            print(f"  Q: How did {filing['ticker']} perform financially?")
            print(f"  Q: Who are {filing['ticker']}'s key executives?")
        else:
            print("\n✗ Pipeline failed")
            print("Check logs at /tmp/sec_intelligence.log")
    
    except Exception as e:
        print("-" * 70)
        print(f"\n✗ Error: {e}")
        print("Check logs at /tmp/sec_intelligence.log")


if __name__ == "__main__":
    main()
