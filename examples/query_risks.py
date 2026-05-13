#!/usr/bin/env python
"""
Example: Get structured risk data directly from database
Run: python examples/query_risks.py
"""

from sec_intelligence import create_platform
from sec_intelligence.storage import get_postgres_db
from sec_intelligence.storage.models import RiskRecord


def main():
    print("\n" + "=" * 70)
    print(" SEC Intelligence Platform - Query Structured Risks")
    print("=" * 70 + "\n")
    
    # Initialize platform
    print("Initializing platform...")
    try:
        platform = create_platform()
        print("✓ Platform ready\n")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return
    
    # Query risks from database
    print("Querying risks from database...")
    print("-" * 70)
    
    try:
        db = get_postgres_db()
        session = db.get_session()
        
        # Get all risks
        risks = session.query(RiskRecord).all()
        
        if not risks:
            print("\nNo risks found in database.")
            print("(Hint: Process a filing first with examples/process_filing.py)")
        else:
            print(f"\nFound {len(risks)} risks:\n")
            
            # Group by severity
            by_severity = {}
            for risk in risks:
                severity = risk.severity
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(risk)
            
            # Display by severity
            severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            for severity in severity_order:
                if severity in by_severity:
                    print(f"\n[{severity}]")
                    for risk in by_severity[severity]:
                        print(f"  • {risk.risk_type}")
                        print(f"    {risk.description[:100]}...")
                        print(f"    Confidence: {risk.confidence:.1%}\n")
        
        session.close()
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
