# Financial Data Multi-Step Analyzer Agent

An intelligent agentic AI system that autonomously conducts comprehensive financial analysis on stock tickers, producing structured investment theses. The agent dynamically decides what data to gather and analyze based on findings, rather than following a fixed script.

## Overview

Tell the agent "analyze Apple," and it will:
- Fetch Apple's financial data and metrics
- Identify and compare to peer companies
- Analyze valuation gaps and competitive positioning
- Detect anomalies and red flags
- Generate a structured investment thesis with recommendation

**Example:** "Analyze AAPL and compare to Microsoft. What's the investment case?"
```
→ Agent gathers data → Compares metrics → Identifies insights
→ Final Output: "Apple trades at 28.5x PE vs MSFT at 24.2x. Premium justified 
   by ecosystem moat. BUY with 15% upside to $195."
```

## Key Concepts

### What Makes This "Agentic"?

**Traditional Pipeline (Fixed sequence):**
```
fetch_data() → compare() → detect_anomalies() → generate_thesis()
```

**Agentic Approach (Adaptive reasoning):**
```
Observe user request
  ↓
Plan: "What data matters most?"
  ↓
Execute: Call tools intelligently
  ↓
Reflect: "Do I need more data?"
  ↓
Iterate: Loop until confident
  ↓
Output: Final thesis with reasoning
```

The agent reasons about what matters, adapts based on findings, and makes decisions autonomously—no fixed script.

---

## How It Works

### Step 1: User Input
```
User: "Evaluate Tesla. Compare to traditional automakers (Ford, GM, VW)."
```

### Step 2: Agent Decides What Tools to Call
Claude analyzes the request and determines:
- ✓ Fetch Tesla's stock data (price, PE, margins, growth)
- ✓ Fetch competitor data (Ford, GM, VW)
- ✓ Compare valuation and growth metrics
- ✓ Detect anomalies and risks
- ✓ Generate investment thesis

### Step 3: Tool Execution

**Tool 1: Fetch Stock Data**
```json
{
  "ticker": "TSLA",
  "current_price": 242.50,
  "market_cap": 768_000_000_000,
  "pe_ratio": 68.2,
  "revenue_growth": 0.25,
  "gross_margin": 0.268,
  "net_debt": 8_200_000_000
}
```

**Tool 2: Fetch Competitors**
```json
{
  "F": {"pe_ratio": 4.2, "revenue_growth": -0.08, "gross_margin": 0.12},
  "GM": {"pe_ratio": 5.1, "revenue_growth": 0.02, "gross_margin": 0.14}
}
```

**Tool 3: Compare Metrics**
```json
{
  "valuation_gap": "Tesla trades 14x peers",
  "growth_gap": "TSLA: +25% YoY vs peers: -1%",
  "margin_analysis": "TSLA: 26.8% vs peers: 13% (2x advantage)"
}
```

**Tool 4: Detect Anomalies**
```json
[
  "P/E 14x above sector average",
  "Debt increased 12% YoY",
  "Gross margin compressed 80bps last quarter",
  "Delivery growth slowed to 6% vs 25% historical"
]
```

### Step 4: Agent Iteration
After initial analysis, the agent may realize it needs more context:
- "Margin compression is concerning—is this temporary or structural?" → Fetch quarterly trends
- "Debt level is rising—is it funding growth or problematic?" → Analyze capital allocation
- Loops until confident enough to generate final thesis

### Step 5: Final Output
```json
{
  "ticker": "TSLA",
  "current_price": 242.50,
  "valuation_assessment": "EXPENSIVE vs traditional auto; FAIR vs growth rate",
  "bull_case": [
    "Only profitable EV maker at scale",
    "Gross margins 2x peers despite price wars",
    "FSD could unlock $100B+ TAM",
    "Energy business growing 50% YoY"
  ],
  "bear_case": [
    "Valuation leaves no room for error",
    "Delivery growth decelerating",
    "Competition from BYD and legacy OEMs",
    "Sentiment-dependent, vulnerable to macro shifts"
  ],
  "key_risks": [
    "Regulatory risk (FSD approval)",
    "Execution risk (new factories)",
    "Demand destruction in China"
  ],
  "recommendation": "HOLD",
  "price_target_12m": "280-320",
  "confidence": 0.72
}
```

---

## Data Flow Architecture

```
┌──────────────────────────────────────┐
│ User: "Is Tesla a good buy?"         │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────┐
│ Claude Agent (Orchestrator)              │
│ • Reads input                            │
│ • Decides tool order & prioritization    │
│ • Manages reasoning state                │
└──────────────────────────────────────────┘
    ↓             ↓              ↓
┌─────────┐  ┌──────────┐  ┌──────────┐
│Fetch    │  │Fetch     │  │Compare   │
│TSLA     │  │Peers     │  │Metrics   │
└─────────┘  └──────────┘  └──────────┘
    ↓             ↓              ↓
 yfinance    yfinance ×3    Python calc
 SEC EDGAR   SEC EDGAR      P/E, margins
    ↓             ↓              ↓
────────────────────────────────────
         ↓
┌────────────────────────────────┐
│ Claude Reflects on Data        │
│ "Margins compressed but still  │
│  2x peers. Growth slowing but  │
│  peers declining. FSD upside?" │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│ Detect Anomalies               │
│ Generate Investment Thesis     │
└────────────────────────────────┘
         ↓
┌────────────────────────────────┐
│ Final Recommendation            │
│ {recommendation, thesis,        │
│  risks, confidence}             │
└────────────────────────────────┘
```

---

## Learning Outcomes

By building this project, you'll master:

| Concept | What You Learn | Why It Matters |
|---------|---|---|
| **Tool Calling** | Define schemas, watch Claude decide which tools to invoke | Core of modern agentic AI |
| **Multi-Turn Reasoning** | Manage message history, watch Claude build on prior data | How agents think iteratively |
| **Error Handling** | Deal with API failures, missing data, parsing errors | Real-world production challenges |
| **State Management** | Track gathered data, decide if more is needed | Essential for complex agents |
| **Structured Output** | Prompt for JSON, ensure consistency | Building reliable AI systems |
| **API Integration** | Wire yfinance, SEC EDGAR, Alpha Vantage | Practical data engineering |
| **Prompt Engineering** | Write clear instructions for tool use, output format | The "art" of working with LLMs |

---

## Project Structure

```
Agentic AI/
├── README.md                 # This file
├── main.py                   # Agent orchestrator
├── tools/
│   ├── fetch_stock_data.py   # Pull financials via yfinance
│   ├── fetch_competitors.py  # Identify & fetch peer data
│   ├── compare_metrics.py    # Valuation & growth analysis
│   ├── detect_anomalies.py   # Red flag detection
│   └── generate_thesis.py    # Synthesize investment view
├── utils/
│   ├── llm_client.py         # Claude API wrapper
│   ├── data_cache.py         # Cache API responses
│   └── formatting.py         # JSON / text utilities
└── config.py                 # API keys, model settings
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- Anthropic API key
- Financial data API keys (yfinance is free, optional: Alpha Vantage, SEC EDGAR)

### Installation

```bash
# Clone repository
git clone https://github.com/xxorks/agentic-ai.git
cd agentic-ai

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ANTHROPIC_API_KEY="your-api-key"
export ALPHA_VANTAGE_KEY="optional-key"
```

### Usage

```bash
# Analyze a single stock
python main.py --ticker TSLA

# Analyze with peer comparison
python main.py --ticker AAPL --compare MSFT GOOGL

# Verbose mode (see agent reasoning)
python main.py --ticker NVDA --verbose
```

---

## Example Output

```
Analyzing TSLA...

Agent Step 1: Fetching Tesla stock data
✓ Current price: $242.50 | Market cap: $768B | P/E: 68.2

Agent Step 2: Fetching competitors (Ford, GM, VW)
✓ Ford P/E: 4.2 | GM P/E: 5.1 | VW P/E: 8.7

Agent Step 3: Comparing metrics
⚠ Valuation gap: Tesla 14x peers—justified by growth?
✓ Growth: Tesla +25% YoY vs peers -1% average

Agent Step 4: Detecting anomalies
⚠ Debt increased 12% YoY—funding or concern?
⚠ Margin compressed 80bps—temporary or structural?

Agent Step 5: Generating thesis
✓ Thesis complete | Confidence: 72%

═══════════════════════════════════════
INVESTMENT THESIS: TSLA
═══════════════════════════════════════
Recommendation: HOLD
Price Target (12M): $280-320
Confidence: 72%

Bull Case:
  • Only profitable EV maker at scale
  • Gross margins 2x peers despite price wars
  • FSD could unlock $100B+ TAM

Bear Case:
  • Valuation leaves no room for error
  • Delivery growth decelerating
  • Increasing competition

Key Risks:
  • Regulatory risk (FSD approval)
  • Execution risk (new factories)
```

---

## Contributing

Contributions welcome! Areas for improvement:
- Additional financial metrics (FCF, ROIC, debt ratios)
- Sentiment analysis (earnings call transcripts, news)
- Sector rotation logic
- Alternative data sources (proprietary datasets)
- Backtesting framework

---

## License

MIT License - See LICENSE file for details

---

## Author

**xxorks** — Building agentic AI for better investment decisions

---

## Disclaimer

This tool is for **educational purposes only**. It is not investment advice. Always consult with a financial advisor before making investment decisions. Past performance does not guarantee future results.