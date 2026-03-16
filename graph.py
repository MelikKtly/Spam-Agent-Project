from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.agents import main_agent, test_agent


class AgentState(TypedDict):
    text: str
    result: str
    validation: str


def run_main_agent(state: AgentState):

    result = main_agent(state["text"])

    return {
        "result": result
    }


def run_test_agent(state: AgentState):

    validation = test_agent(
        state["text"],
        state["result"]
    )

    return {
        "validation": validation
    }


def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("main_agent", run_main_agent)
    workflow.add_node("test_agent", run_test_agent)

    workflow.set_entry_point("main_agent")

    workflow.add_edge("main_agent", "test_agent")
    workflow.add_edge("test_agent", END)

    return workflow.compile()