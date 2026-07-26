"""
Sub-agents you want to expose as tools to your root agent.

Each `Agent` defined here can be wrapped in `AgentTool` and added to
`root_agent.tools` in `agent.py`. Sub-agents are useful for delegating
specialized work (e.g. a separate prompt + model for web search, a
classifier, a translator) without polluting the root agent's prompt.

Pattern:

    from google.adk.agents import Agent
    from google.adk.tools import google_search
    import os

    google_search_agent = Agent(
        model=os.environ.get('QUICK_AGENT_MODEL', 'gemini-3-flash-preview'),
        name='google_search_agent',
        description='Performs Google searches and returns relevant results.',
        instruction='Search the web for the user query and return the most relevant results.',
        tools=[google_search],
    )

Then in agent.py:

    from google.adk.tools.agent_tool import AgentTool
    from .custom_agents import google_search_agent
    ...
    tools=[AgentTool(agent=google_search_agent)],
"""
