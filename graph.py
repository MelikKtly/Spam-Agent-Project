from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from agents.agents import main_agent, test_agent


class AgentState(TypedDict):
    text: str
    result: str
    validation: str



def run_main_agent(state: AgentState) -> AgentState:
    result = main_agent(state['text'])
    return {
        'result': result
    }

def run_test_agent(state: AgentState) -> AgentState:
    validation = test_agent(state['text'], state['result'])
    return {
        'validation': validation
    }

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("main_agent", run_main_agent)
    graph.add_node("test_agent", run_test_agent)

    graph.add_edge(START, "main_agent")
    graph.add_edge("main_agent", "test_agent")
    graph.add_edge("test_agent", END)

    return graph.compile

