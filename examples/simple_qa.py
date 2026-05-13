#!/usr/bin/env python
"""
Simple example: Ask questions about SEC filings
Run: python examples/simple_qa.py
"""

from sec_intelligence import create_platform

def main():
    print("=" * 60)
    print("SEC Intelligence Platform - Simple Q&A Example")
    print("=" * 60)
    
    # Initialize platform
    print("\n1. Initializing platform...")
    try:
        platform = create_platform()
        print("   ✓ Platform initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # Example questions
    questions = [
        "What are the main business risks?",
        "How is the company structured?",
        "What are the key financial metrics?",
    ]
    
    # Ask questions (without real data, these will return limited results)
    print("\n2. Asking questions...")
    
    for question in questions:
        print(f"\n   Q: {question}")
        try:
            response = platform.ask_question(
                question,
                ticker="NVDA",
                filing_types=["10-K"]
            )
            print(f"   A: {response.answer[:200]}...")
            print(f"   Confidence: {response.confidence:.1%}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
