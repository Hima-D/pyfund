# src/pyfundlib/ml/interpretive.py
from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from pyfundlib.config import settings
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


class GenAIInterpreter:
    """
    Translates complex quant metrics into human insights.
    Eliminates hallucinations by grounding prompts in real computed data.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        # This can be configured for Grok-4 or other 2026 models via environment
        api_key = settings.api_key
        base_url = None
        
        # Heuristic for Grok/Llama if configured
        if "grok" in model_name.lower():
            base_url = "https://api.x.ai/v1"
            
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,  # Low temperature for deterministic analysis
        )

    def explain_risk(self, portfolio_data: dict[str, Any]) -> str:
        """
        Grounded risk interpretation.
        Input is a dictionary of metrics computed by PortfolioAllocator or RiskEngine.
        """
        prompt = (
            "You are an elite institutional risk manager. Explain the following portfolio risk metrics "
            "to a Portfolio Manager. Use professional, concise language. Be specific about the numbers. "
            "Highlight anomalies or high concentration risks.\n\n"
            f"Metrics: {portfolio_data}"
        )
        
        try:
            logger.info("explaining_risk", tickers=len(portfolio_data.get("weights", {})))
            response = self.llm.invoke([
                SystemMessage(content="You transcribe quant data into high-conviction risk summaries. Cite the data exactly."),
                HumanMessage(content=prompt)
            ])
            return str(response.content)
        except Exception as e:
            logger.error("genai_interpretation_failed", error=str(e))
            return f"Interpretation unavailable: {e}"

    def explain_trade_signal(self, ticker: str, technicals: dict[str, Any], ml_score: float) -> str:
        """Explain why a strategy is triggering a signal"""
        prompt = (
            f"Explain the {ticker} signal. Technicals: {technicals}. ML Confidence: {ml_score:.2f}. "
            "Synthesize these into a 2-sentence rationale."
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return str(response.content)
        except Exception as e:
            return f"Signal rationale missing: {e}"
