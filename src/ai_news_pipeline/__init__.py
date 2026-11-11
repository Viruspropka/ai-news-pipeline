"""ai-news-pipeline: ingest, dedupe, score, verify and brief AI news with an LLM.

Every model call goes through one adapter (`ai_news_pipeline.llm`), and a
deterministic mock backend lets the whole pipeline run offline.
"""

__version__ = "0.1.0"
