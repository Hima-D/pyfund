# src/pyfundlib/agents/orchestrator.py
from __future__ import annotations

import os
from typing import Any, Optional

from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

from pyfundlib.config import settings
from .data_agent import FetchPriceTool, ResampleDataTool
from .risk_agent import ComputeVaRTool
from pyfundlib.utils.logger import get_logger

logger = get_logger(__name__)


class QuantCrew:
    """
    Orchestrates a swarm of quant agents for automated research.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.api_key,
            temperature=0,
        )

        # 1. Define Agents
        self.data_agent = Agent(
            role="Data Engineer",
            goal="Fetch and clean requested financial data accurately.",
            backstory="You are a meticulous data engineer specialized in high-frequency financial data.",
            tools=[FetchPriceTool(), ResampleDataTool()],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        self.risk_agent = Agent(
            role="Risk Manager",
            goal="Identify and quantify portfolio risks.",
            backstory="You are an expert risk manager who formerly worked at a major hedge fund.",
            tools=[ComputeVaRTool()],
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        self.research_agent = Agent(
            role="Quant Researcher",
            goal="Synthesize data and risk insights into a trade strategy.",
            backstory="You are a world-class quantitative researcher known for alpha generation.",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
        )

    def analyze_ticker(self, ticker: str) -> str:
        """Run a full agentic analysis on a ticker."""
        logger.info("agent_run_started", ticker=ticker)
        
        # 2. Define Tasks
        fetch_task = Task(
            description=f"Fetch the last year of daily data for {ticker} and summarize it.",
            expected_output="A summary of the price action and data quality.",
            agent=self.data_agent,
        )

        risk_task = Task(
            description=f"Analyze the risk profile of {ticker} using VaR analysis.",
            expected_output="A detailed risk assessment report.",
            agent=self.risk_agent,
        )

        synthesis_task = Task(
            description=f"Based on the price data and risk profile of {ticker}, provide a final recommendation.",
            expected_output="A concise investment recommendation memo.",
            agent=self.research_agent,
            context=[fetch_task, risk_task],
        )

        # 3. Assemble Crew
        crew = Crew(
            agents=[self.data_agent, self.risk_agent, self.research_agent],
            tasks=[fetch_task, risk_task, synthesis_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        logger.info("agent_run_completed", ticker=ticker)
        return str(result)
