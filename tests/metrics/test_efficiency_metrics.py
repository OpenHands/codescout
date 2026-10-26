"""Tests for efficiency metrics computation."""

import pytest
from src.metrics.efficiency_metrics import (
    compute_token_metrics,
    compute_step_count,
    compute_tool_call_metrics,
    compute_all_efficiency_metrics,
)


# Mock data structures
MOCK_TOKEN_MESSAGE_1 = {
    "kind": "TokenEvent",
    "prompt_token_ids": [1, 2, 3, 4],  # 4 tokens
    "response_token_ids": [5, 6, 7],  # 3 tokens
}

MOCK_TOKEN_MESSAGE_2 = {
    "kind": "TokenEvent",
    "prompt_token_ids": [1, 2, 3, 4, 5, 6],  # 6 tokens
    "response_token_ids": [7, 8],  # 2 tokens
}

MOCK_ASSISTANT_MESSAGE_WITH_TOOLS = {
    "role": "assistant",
    "content": "I'll search for the file",
    "tool_calls": [
        {
            "function": {
                "name": "bash",
                "arguments": '{"command": "rg pattern"}'
            }
        },
        {
            "function": {
                "name": "result",
                "arguments": '{"file_paths": ["file.py"]}'
            }
        }
    ]
}

MOCK_ASSISTANT_MESSAGE_NO_TOOLS = {
    "role": "assistant",
    "content": "Here's the result",
}

MOCK_TOOL_RESPONSE = {
    "role": "tool",
    "content": "Search results here",
}

MOCK_ACTION_EVENT_BASH = {
    "kind": "ActionEvent",
    "tool_name": "bash",
    "llm_response_id": "turn-1",
}

MOCK_ACTION_EVENT_RESULT = {
    "kind": "ActionEvent",
    "tool_name": "result",
    "llm_response_id": "turn-1",
}

MOCK_ACTION_EVENT_UNKNOWN = {
    "kind": "ActionEvent",
    "llm_response_id": "turn-1",
}


class TestComputeTokenMetrics:
    """Tests for compute_token_metrics function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        result = compute_token_metrics([])
        assert result["total_tokens"] == 0
        assert result["total_prompt_tokens"] == 0
        assert result["total_response_tokens"] == 0
        assert result["avg_prompt_tokens_per_step"] == 0.0
        assert result["avg_response_tokens_per_step"] == 0.0

    def test_no_token_events(self):
        """Test with messages but no TokenEvents."""
        messages = [MOCK_ASSISTANT_MESSAGE_NO_TOOLS, MOCK_TOOL_RESPONSE]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 0

    def test_single_token_event(self):
        """Test with single TokenEvent."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 7  # 4 prompt + 3 response
        assert result["total_prompt_tokens"] == 4
        assert result["total_response_tokens"] == 3
        assert result["avg_prompt_tokens_per_step"] == 4.0
        assert result["avg_response_tokens_per_step"] == 3.0

    def test_multiple_token_events(self):
        """Test with multiple TokenEvents."""
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_TOKEN_MESSAGE_2]
        result = compute_token_metrics(messages)
        assert result["total_tokens"] == 11  # latest prompt + all responses
        assert result["total_prompt_tokens"] == 6
        assert result["total_response_tokens"] == 5  # 3 + 2
        assert result["avg_prompt_tokens_per_step"] == 3.0  # 6/2
        assert result["avg_response_tokens_per_step"] == 2.5  # 5/2

    def test_mixed_messages(self):
        """Test with mix of TokenEvents and other messages."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ASSISTANT_MESSAGE_NO_TOOLS,
            MOCK_TOKEN_MESSAGE_2,
            MOCK_TOOL_RESPONSE,
        ]
        result = compute_token_metrics(messages)
        # Should only count the two TokenEvents
        assert result["total_tokens"] == 11
        assert result["total_prompt_tokens"] == 6
        assert result["total_response_tokens"] == 5


class TestComputeStepCount:
    """Tests for compute_step_count function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        assert compute_step_count([]) == 0

    def test_no_token_events(self):
        """Test with messages but no TokenEvents."""
        messages = [MOCK_ASSISTANT_MESSAGE_NO_TOOLS, MOCK_TOOL_RESPONSE]
        assert compute_step_count(messages) == 0

    def test_single_token_event(self):
        """Test with single TokenEvent."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        assert compute_step_count(messages) == 1

    def test_multiple_token_events(self):
        """Test with multiple TokenEvents."""
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_TOKEN_MESSAGE_2]
        assert compute_step_count(messages) == 2

    def test_mixed_messages(self):
        """Test with mix of TokenEvents and other messages."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ASSISTANT_MESSAGE_NO_TOOLS,
            MOCK_TOKEN_MESSAGE_2,
            MOCK_TOOL_RESPONSE,
        ]
        assert compute_step_count(messages) == 2


class TestComputeToolCallMetrics:
    """Tests for compute_tool_call_metrics function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        result = compute_tool_call_metrics([])
        assert result["total_tool_calls"] == 0
        assert result["avg_tool_calls_per_step"] == 0.0
        assert result["tool_call_breakdown"] == {}

    def test_no_tool_calls(self):
        """Test with messages but no tool calls."""
        messages = [MOCK_ASSISTANT_MESSAGE_NO_TOOLS, MOCK_TOOL_RESPONSE, MOCK_TOKEN_MESSAGE_1]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 0
        assert result["avg_tool_calls_per_step"] == 0.0

    def test_single_assistant_message_with_tools(self):
        """Test with single assistant message containing tool calls."""
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_ASSISTANT_MESSAGE_WITH_TOOLS]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 2
        assert result["avg_tool_calls_per_step"] == 2.0  # 2 tools / 1 step
        assert result["tool_call_breakdown"]["bash"] == 1
        assert result["tool_call_breakdown"]["result"] == 1

    def test_multiple_tool_calls(self):
        """Test with multiple assistant messages with tools."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ASSISTANT_MESSAGE_WITH_TOOLS,
            MOCK_TOKEN_MESSAGE_2,
            MOCK_ASSISTANT_MESSAGE_WITH_TOOLS,
        ]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 4  # 2 tools x 2 messages
        assert result["avg_tool_calls_per_step"] == 2.0  # 4 tools / 2 steps
        assert result["tool_call_breakdown"]["bash"] == 2
        assert result["tool_call_breakdown"]["result"] == 2

    def test_mixed_assistant_messages(self):
        """Test with mix of assistant messages with and without tools."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ASSISTANT_MESSAGE_WITH_TOOLS,  # 2 tools
            MOCK_ASSISTANT_MESSAGE_NO_TOOLS,  # 0 tools
            MOCK_TOKEN_MESSAGE_2,
        ]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 2
        assert result["avg_tool_calls_per_step"] == 1.0  # 2 tools / 2 steps

    def test_action_event_tool_calls(self):
        """Test tool counting from SDK ActionEvent trajectory messages."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,
            MOCK_ACTION_EVENT_BASH,
            MOCK_ACTION_EVENT_RESULT,
        ]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 2
        assert result["avg_tool_calls_per_step"] == 2.0
        assert result["tool_call_breakdown"]["bash"] == 1
        assert result["tool_call_breakdown"]["result"] == 1

    def test_action_event_without_tool_name(self):
        """Test ActionEvent fallback when tool_name is absent."""
        messages = [MOCK_TOKEN_MESSAGE_1, MOCK_ACTION_EVENT_UNKNOWN]
        result = compute_tool_call_metrics(messages)
        assert result["total_tool_calls"] == 1
        assert result["tool_call_breakdown"]["unknown"] == 1


class TestComputeAllEfficiencyMetrics:
    """Tests for compute_all_efficiency_metrics function."""

    def test_empty_messages(self):
        """Test with empty messages list."""
        result = compute_all_efficiency_metrics(
            messages=[],
            wall_clock_duration=10.5,
        )
        assert result["tokens"] == 0
        assert result["steps"] == 0
        assert result["avg_tool_calls_per_step"] == 0.0
        assert result["wall_clock_duration"] == 10.5

    def test_complete_trajectory(self):
        """Test with complete trajectory including tokens, steps, and tools."""
        messages = [
            MOCK_TOKEN_MESSAGE_1,  # Step 1: 7 tokens
            MOCK_ACTION_EVENT_BASH,
            MOCK_ACTION_EVENT_RESULT,
            MOCK_TOKEN_MESSAGE_2,  # Step 2: 8 tokens
            MOCK_ACTION_EVENT_BASH,
            MOCK_ACTION_EVENT_RESULT,
        ]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=15.5,
            start_timestamp="2025-01-01T10:00:00",
            end_timestamp="2025-01-01T10:00:15",
        )

        # Check core metrics
        assert result["tokens"] == 11  # latest prompt + all responses
        assert result["steps"] == 2
        assert result["avg_tool_calls_per_step"] == 2.0  # 4 tools / 2 steps
        assert result["wall_clock_duration"] == 15.5

        # Check flattened token fields
        assert result["total_prompt_tokens"] == 6
        assert result["total_response_tokens"] == 5
        assert result["avg_prompt_tokens_per_step"] == 3.0
        assert result["avg_response_tokens_per_step"] == 2.5

    def test_without_timestamps(self):
        """Test that timestamps are optional."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=5.0,
        )
        assert "start_timestamp" not in result
        assert "end_timestamp" not in result

    def test_minimal_trajectory(self):
        """Test with minimal valid trajectory (single step, no tools)."""
        messages = [MOCK_TOKEN_MESSAGE_1]
        result = compute_all_efficiency_metrics(
            messages=messages,
            wall_clock_duration=2.0,
        )
        assert result["tokens"] == 7
        assert result["steps"] == 1
        assert result["avg_tool_calls_per_step"] == 0.0
        assert result["wall_clock_duration"] == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
