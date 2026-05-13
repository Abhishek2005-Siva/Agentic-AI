#!/bin/bash
# Quick run script for SEC Intelligence Platform

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to script directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

echo -e "${BLUE}SEC Intelligence Platform${NC}"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""
echo "Available commands:"
echo "  python examples/interactive_qa.py      - Interactive Q&A mode"
echo "  python examples/simple_qa.py           - Simple Q&A example"
echo "  python examples/process_filing.py      - Process a filing"
echo "  python examples/query_risks.py         - Query extracted risks"
echo ""
echo "Quick test:"
python3 -c "from sec_intelligence import create_platform; print('✓ Platform ready to use')"
echo ""
echo "Now run any of the above commands, e.g.:"
echo "  python examples/interactive_qa.py"
