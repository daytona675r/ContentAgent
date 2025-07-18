import pytest

from agent import contentGraph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {"changeme": "some_val"}
    res = await contentGraph.ainvoke(inputs)
    assert res is not None
