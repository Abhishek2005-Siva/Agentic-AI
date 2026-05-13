#!/usr/bin/env python
"""
Interactive example: Real-time Q&A with user input
Run: python examples/interactive_qa.py
"""

from sec_intelligence import create_platform


def main():
    print("\n" + "=" * 70)
    print(" SEC Intelligence Platform - Interactive Q&A")
    print("=" * 70)
    print("\nInitializing platform...", end=" ")
    
    try:
        platform = create_platform()
        print("✓\n")
    except Exception as e:
        print(f"✗\nError: {e}")
        return
    
    print("Type 'help' for commands, 'exit' to quit\n")
    
    # Get ticker
    ticker = input("Enter stock ticker (e.g., NVDA, TSLA, MSFT): ").strip().upper()
    if not ticker:
        ticker = "NVDA"
    
    print(f"\nAsking about: {ticker}")
    print("-" * 70)
    
    # Interactive loop
    while True:
        try:
            question = input(f"\n[{ticker}] Ask: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            if question.lower() == 'help':
                print_help()
                continue
            
            if question.lower() == 'ticker':
                ticker = input("New ticker: ").strip().upper()
                print(f"Now asking about: {ticker}")
                continue
            
            # Ask the question
            print("\nSearching... ", end="", flush=True)
            
            response = platform.ask_question(
                question,
                ticker=ticker,
                filing_types=["10-K", "10-Q"]
            )
            
            print("Done!")
            print("-" * 70)
            print(f"\nAnswer:\n{response.answer}\n")
            print(f"Confidence: {response.confidence:.1%}")
            print(f"Supporting chunks: {len(response.supporting_chunks)}")
            
            # Show first supporting chunk
            if response.supporting_chunks:
                chunk = response.supporting_chunks[0]
                print(f"\nTop supporting evidence ({chunk.chunk.section}):")
                print(f"  {chunk.chunk.text[:150]}...")
            
            print("-" * 70)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def print_help():
    print("\n" + "-" * 70)
    print("Commands:")
    print("  help     - Show this help message")
    print("  ticker   - Change stock ticker")
    print("  exit     - Exit the program")
    print("\nExample questions:")
    print("  'What are the main risks?'")
    print("  'How much revenue did they make?'")
    print("  'Who are the executives?'")
    print("  'What acquisitions did they make?'")
    print("  'What is their geographic breakdown?'")
    print("-" * 70)


if __name__ == "__main__":
    main()
