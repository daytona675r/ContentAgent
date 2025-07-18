from langgraph.pregel import Pregel

from agent.contentGraph import graph as contentGraph


def test_placeholder() -> None:
    # TODO: You can add actual unit tests
    # for your graph and other logic here.
    assert isinstance(contentGraph, Pregel)
