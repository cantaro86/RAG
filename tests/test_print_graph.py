from unittest.mock import MagicMock

import pytest

from scripts import print_graph

pytestmark = pytest.mark.cpu


def test_build_graph_for_rendering_does_not_invoke_runtime_dependencies(config_data, monkeypatch):
    dependency = MagicMock()
    dependency_factory = MagicMock(return_value=dependency)
    monkeypatch.setattr(print_graph, "MagicMock", dependency_factory)

    graph = print_graph._build_graph_for_rendering(config_data)

    expected_nodes = {
        "sanitize_question",
        "guardrail",
        "handle_hello",
        "handle_thanks",
        "handle_off_topic",
        "init_first_question",
        "topic_detector",
        "pre_retrieval_rewriter",
        "clear_history",
        "retrieve_and_filter",
        "transform_query",
        "generate_with_docs",
        "clean_answer",
        "no_generation",
    }
    assert expected_nodes <= set(graph.nodes)
    dependency_factory.assert_called_once_with(name="unloaded_graph_dependency")
    assert dependency.mock_calls == []
