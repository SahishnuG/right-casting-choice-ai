from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
from .tools.omdb import OmdbTool
import os
import google.generativeai as genai
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class RightCastingChoiceAi():
    """RightCastingChoiceAi crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def character_extractor(self) -> Agent:
        # Gemini setup from env
        gemini_key = (
            os.getenv("GOOGLE_API_KEY")
        )
        if gemini_key:
            genai.configure(api_key=gemini_key)
        return Agent(
            config=self.agents_config['character_extractor'], # type: ignore[index]
            llm="gemini/gemini-2.5-flash",
            verbose=True
        )

    @agent
    def similar_movies_and_omdb(self) -> Agent:
        # Provide Serper tool and OMDb wrapper via agent tools
        omdb_key = os.getenv("OMDB_API_KEY", "")
        omdb_tool = OmdbTool()
        tools = [SerperDevTool(), omdb_tool]
        return Agent(
            config=self.agents_config['similar_movies_and_omdb'], # type: ignore[index]
            tools=tools,
            llm="gemini/gemini-2.5-flash",
            verbose=True
        )

    @agent
    def budget_ranker(self) -> Agent:
        return Agent(
            config=self.agents_config['budget_ranker'], # type: ignore[index]
            llm="gemini/gemini-2.5-flash",
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def extract_characters_task(self) -> Task:
        return Task(
            config=self.tasks_config['extract_characters_task'], # type: ignore[index]
        )

    @task
    def similar_movies_task(self) -> Task:
        return Task(
            config=self.tasks_config['similar_movies_task'], # type: ignore[index]
        )

    @task
    def rank_candidates_task(self) -> Task:
        return Task(
            config=self.tasks_config['rank_candidates_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RightCastingChoiceAi crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
