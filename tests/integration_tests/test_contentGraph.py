from langgraph.pregel import Pregel

from agent.contentGraph import graph as contentGraph


def test_legacy_graph_compiles_without_provider_calls() -> None:
    assert isinstance(contentGraph, Pregel)
