"""
Example usage of SEC Intelligence Platform
Run this to see the system in action
"""

import sys
import logging
from sec_intelligence import create_platform
from sec_intelligence.schemas import FilingType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_setup():
    """Example 1: Basic platform setup and initialization"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Platform Setup")
    print("="*70)
    
    try:
        platform = create_platform()
        print("✅ Platform initialized successfully")
        print(f"   Config: {platform.config.app_name} v{platform.config.app_version}")
    except Exception as e:
        print(f"❌ Failed to initialize platform: {e}")
        print("   Make sure PostgreSQL and Qdrant are running")
        return False
    
    return True


def example_2_process_filing():
    """Example 2: Process a SEC filing"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Process a SEC Filing")
    print("="*70)
    
    platform = create_platform()
    
    # Note: This is a mock example. In real usage:
    # 1. Get actual CIK from SEC search
    # 2. Get actual accession number from SEC EDGAR
    # 3. Filing will be fetched from SEC servers
    
    print("\nProcessing NVIDIA 10-K filing...")
    print("Steps:")
    print("  1. Ingestion Agent - Fetch from SEC EDGAR")
    print("  2. Structure Agent - Detect sections (Item 1, 1A, 7, 8...)")
    print("  3. Extraction Agents - Extract risks, metrics, entities, events")
    print("  4. Validation Agent - Verify completeness")
    print("  5. Store in PostgreSQL + Qdrant")
    
    # In production:
    # success = platform.process_filing(
    #     ticker="NVDA",
    #     cik="0001045810",
    #     accession_number="0001047469-25-001234",
    #     filing_type="10-K"
    # )
    
    print("\n✅ Pipeline structure is ready (execution requires real SEC data)")


def example_3_question_answering():
    """Example 3: Question answering"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Question Answering")
    print("="*70)
    
    platform = create_platform()
    
    example_questions = [
        "What are the main risks NVIDIA faces?",
        "How did revenue change year-over-year?",
        "Who are the key executives?",
        "What major acquisitions happened?",
        "What is the gross profit margin trend?",
    ]
    
    print("\nExample questions that can be answered:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    print("\nAnswer Flow:")
    print("  1. Retrieval Agent - Decide retrieval strategy")
    print("  2. Hybrid Retriever - BM25 + vector search")
    print("  3. Reranker - Cross-encoder refines results")
    print("  4. Compressor - Compress for LLM token budget")
    print("  5. Claude - Synthesize answer from context")
    
    # In production:
    # response = platform.ask_question(
    #     "What are the main cybersecurity risks?",
    #     ticker="NVDA",
    #     filing_types=["10-K"]
    # )
    # print(f"Answer: {response.answer}")
    # print(f"Confidence: {response.confidence:.2%}")


def example_4_comparison():
    """Example 4: Filing comparison"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Filing Comparison")
    print("="*70)
    
    platform = create_platform()
    
    print("\nComparing NVIDIA 10-K filings across 3 years:")
    print("\nComparison includes:")
    print("  ✓ New risks vs prior year")
    print("  ✓ Risk severity changes")
    print("  ✓ Financial metric trends")
    print("  ✓ Revenue/margin evolution")
    print("  ✓ Key structural changes")
    print("  ✓ New entities (acquisitions)")
    
    # In production:
    # result = platform.compare_filings(
    #     ticker="NVDA",
    #     years=[2023, 2024, 2025],
    #     focus_areas=["cybersecurity", "supply_chain", "revenue"]
    # )
    # print(f"\nKey changes:\n{result.key_changes}")


def example_5_structured_queries():
    """Example 5: Structured intelligence queries"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Structured Intelligence Queries")
    print("="*70)
    
    platform = create_platform()
    
    print("\nStructured Queries (no LLM needed, pure SQL):")
    print("\n1. Get all risks for a company:")
    print("   platform.get_company_risks('NVDA', year=2025)")
    print("   → Returns: RiskRecord[] with severity, type, confidence")
    
    print("\n2. Get financial metrics across years:")
    print("   platform.get_financial_metrics('NVDA', years=[2023,2024,2025])")
    print("   → Returns: MetricRecord[] with trends, YoY changes")
    
    print("\n3. Direct SQL queries possible:")
    print("   - Show all companies with AI regulation risks")
    print("   - Find cybersecurity risks > confidence 0.8")
    print("   - Compare gross margins across tech companies")
    
    print("\n✅ Structured queries are ~100x faster than RAG")


def example_6_architecture_overview():
    """Example 6: Architecture overview"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Architecture Overview")
    print("="*70)
    
    print("""
5-Layer SEC Intelligence Platform:

┌─────────────────────────────────────────────────────────────┐
│ Layer 5: AGENTIC REASONING                                   │
│ - Agent orchestrator                                         │
│ - Question answering                                         │
│ - Filing comparison                                          │
│ - Multi-step workflows                                       │
└─────────────────────────────────────────────────────────────┘
                             ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: RETRIEVAL                                            │
│ - Hybrid search (BM25 + vectors)                             │
│ - Reranking (cross-encoder)                                  │
│ - Compression                                                 │
│ - Metadata filtering                                         │
└─────────────────────────────────────────────────────────────┘
                             ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: INTELLIGENCE EXTRACTION                             │
│ - Risk extraction agent                                      │
│ - Financial metrics agent                                    │
│ - Entity extraction agent                                    │
│ - Event detection agent                                      │
└─────────────────────────────────────────────────────────────┘
                             ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: STRUCTURING                                          │
│ - Section detection                                          │
│ - Semantic chunking                                          │
│ - Table extraction                                           │
└─────────────────────────────────────────────────────────────┘
                             ↑
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: INGESTION                                            │
│ - Fetch SEC filings                                          │
│ - Clean HTML                                                 │
│ - Extract metadata                                           │
│ - Store raw documents                                        │
└─────────────────────────────────────────────────────────────┘
                             ↑
                       SEC EDGAR API
    """)
    
    print("Storage Architecture:")
    print("  PostgreSQL: Metadata, chunks, extracted intelligence")
    print("  Qdrant: Vector embeddings for semantic search")
    print("  Object Storage: Raw HTML/JSON filings")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("SEC INTELLIGENCE PLATFORM - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Example 1: Setup
        if not example_1_basic_setup():
            print("\n⚠️  Platform initialization issue detected")
            print("   Make sure you have:")
            print("   - PostgreSQL running on localhost:5432")
            print("   - Qdrant running on localhost:6333")
            print("   - ANTHROPIC_API_KEY environment variable set")
            return
        
        # Run other examples
        example_6_architecture_overview()
        example_2_process_filing()
        example_3_question_answering()
        example_4_comparison()
        example_5_structured_queries()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("""
✅ Architecture Overview: Complete 5-layer system with multi-agents
✅ Filing Processing: Ingestion → Structuring → Intelligence → Storage
✅ Question Answering: Hybrid retrieval → Reranking → LLM synthesis
✅ Filing Comparison: Track risks, metrics, and changes over time
✅ Structured Queries: SQL-based intelligence without RAG

Next Steps:
1. Configure .env with your Anthropic API key
2. Set up PostgreSQL and Qdrant databases
3. Process real SEC filings with platform.process_filing()
4. Ask questions with platform.ask_question()
5. Compare filings with platform.compare_filings()

Documentation:
- README.md: Quick start and overview
- ARCHITECTURE.md: Deep dive into design and implementation
- sec_intelligence/schemas/__init__.py: Data model reference
        """)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
