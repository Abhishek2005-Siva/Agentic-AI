export default {
  name: 'SEC Intelligence',
  repo: 'https://github.com/Abhishek2005-Siva/Agentic-AI',
  eyebrow: 'Agentic financial research',
  tagline: 'An agent that reads SEC EDGAR for you and answers in plain language.',
  description:
    'An autonomous financial intelligence system that queries SEC EDGAR in real time, extracts structured events and entities from filings, and answers natural-language questions through an OpenAI function-calling loop.',
  stack: ['Python', 'OpenAI function calling', 'Gradio', 'SQLite', 'BM25 + embeddings'],
  notice:
    'This is a showcase page. The Gradio app needs an OpenAI API key, so it runs locally from the repository.',
  steps: [
    { title: 'Ask', text: 'Type a question about a company or filing in the Gradio UI.' },
    { title: 'Plan', text: 'The orchestrator lets the model choose which tools to call, and paginates automatically.' },
    { title: 'Fetch and extract', text: 'Pulls filings from EDGAR with rate limiting and caching, then extracts events and entities.' },
    { title: 'Answer', text: 'Loops until the question is fully answered, with readable tool summaries along the way.' },
  ],
  features: [
    { title: 'Real-time EDGAR access', text: 'Automatic rate limiting, retry and caching, with no pre-processing pipeline required.' },
    { title: 'Event extraction', text: 'Classifies appointments, resignations, acquisitions, earnings, dividends, restructurings and legal actions.' },
    { title: 'Entity recognition', text: 'Extracts people, companies, financial figures, dates, locations and tickers.' },
    { title: 'Hybrid local search', text: 'BM25 plus sentence-transformer vectors over filings stored in one local SQLite file.' },
  ],
  startNote: 'Needs Python and an OpenAI API key. Copy .env.example to .env and fill it in.',
  quickstart: `git clone https://github.com/Abhishek2005-Siva/Agentic-AI
cd Agentic-AI
python -m venv venv && source venv/bin/activate
pip install -e .
cp .env.example .env

python app.py                # opens at http://localhost:7860`,
}
