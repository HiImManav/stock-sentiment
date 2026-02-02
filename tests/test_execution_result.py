"""Comprehensive unit tests for orchestrator execution result dataclasses.

Tests for:
- AgentStatus enum
- RouteType enum
- AgentResult dataclass
- QueryClassification dataclass
"""

from __future__ import annotations

import pytest

from src.orchestrator.execution.result import (
    AgentResult,
    AgentStatus,
    QueryClassification,
    RouteType,
)


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_success_value(self) -> None:
        """SUCCESS has correct string value."""
        assert AgentStatus.SUCCESS.value == "success"

    def test_timeout_value(self) -> None:
        """TIMEOUT has correct string value."""
        assert AgentStatus.TIMEOUT.value == "timeout"

    def test_error_value(self) -> None:
        """ERROR has correct string value."""
        assert AgentStatus.ERROR.value == "error"

    def test_all_values_are_unique(self) -> None:
        """All enum values are unique."""
        values = [status.value for status in AgentStatus]
        assert len(values) == len(set(values))

    def test_enum_membership(self) -> None:
        """Can check enum membership."""
        assert AgentStatus.SUCCESS in AgentStatus
        assert AgentStatus.TIMEOUT in AgentStatus
        assert AgentStatus.ERROR in AgentStatus

    def test_enum_from_value(self) -> None:
        """Can create enum from string value."""
        assert AgentStatus("success") == AgentStatus.SUCCESS
        assert AgentStatus("timeout") == AgentStatus.TIMEOUT
        assert AgentStatus("error") == AgentStatus.ERROR

    def test_invalid_value_raises(self) -> None:
        """Invalid value raises ValueError."""
        with pytest.raises(ValueError):
            AgentStatus("invalid")

    def test_enum_comparison(self) -> None:
        """Enum values can be compared."""
        assert AgentStatus.SUCCESS == AgentStatus.SUCCESS
        assert AgentStatus.SUCCESS != AgentStatus.ERROR
        assert AgentStatus.TIMEOUT != AgentStatus.ERROR


class TestRouteType:
    """Tests for RouteType enum."""

    def test_news_only_value(self) -> None:
        """NEWS_ONLY has correct string value."""
        assert RouteType.NEWS_ONLY.value == "news_only"

    def test_sec_only_value(self) -> None:
        """SEC_ONLY has correct string value."""
        assert RouteType.SEC_ONLY.value == "sec_only"

    def test_both_value(self) -> None:
        """BOTH has correct string value."""
        assert RouteType.BOTH.value == "both"

    def test_all_values_are_unique(self) -> None:
        """All enum values are unique."""
        values = [route.value for route in RouteType]
        assert len(values) == len(set(values))

    def test_enum_from_value(self) -> None:
        """Can create enum from string value."""
        assert RouteType("news_only") == RouteType.NEWS_ONLY
        assert RouteType("sec_only") == RouteType.SEC_ONLY
        assert RouteType("both") == RouteType.BOTH

    def test_invalid_value_raises(self) -> None:
        """Invalid value raises ValueError."""
        with pytest.raises(ValueError):
            RouteType("invalid")


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    # -------------------------------------------------------------------------
    # Creation tests
    # -------------------------------------------------------------------------

    def test_create_success_result(self) -> None:
        """Can create a successful agent result."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="Analysis complete",
            execution_time_ms=150.5,
        )

        assert result.agent_name == "news_agent"
        assert result.status == AgentStatus.SUCCESS
        assert result.response == "Analysis complete"
        assert result.error_message is None
        assert result.execution_time_ms == 150.5

    def test_create_timeout_result(self) -> None:
        """Can create a timeout agent result."""
        result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.TIMEOUT,
            error_message="Agent timed out after 60 seconds",
            execution_time_ms=60000.0,
        )

        assert result.agent_name == "sec_agent"
        assert result.status == AgentStatus.TIMEOUT
        assert result.response is None
        assert result.error_message == "Agent timed out after 60 seconds"

    def test_create_error_result(self) -> None:
        """Can create an error agent result."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.ERROR,
            error_message="Connection failed",
            execution_time_ms=50.0,
        )

        assert result.agent_name == "news_agent"
        assert result.status == AgentStatus.ERROR
        assert result.response is None
        assert result.error_message == "Connection failed"

    def test_default_values(self) -> None:
        """Default values are applied correctly."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
        )

        assert result.response is None
        assert result.error_message is None
        assert result.execution_time_ms == 0.0

    def test_agent_name_literal_news(self) -> None:
        """Agent name accepts 'news_agent'."""
        result = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        assert result.agent_name == "news_agent"

    def test_agent_name_literal_sec(self) -> None:
        """Agent name accepts 'sec_agent'."""
        result = AgentResult(agent_name="sec_agent", status=AgentStatus.SUCCESS)
        assert result.agent_name == "sec_agent"

    # -------------------------------------------------------------------------
    # Property tests
    # -------------------------------------------------------------------------

    def test_is_success_true(self) -> None:
        """is_success returns True for SUCCESS status."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
        )
        assert result.is_success is True

    def test_is_success_false_for_timeout(self) -> None:
        """is_success returns False for TIMEOUT status."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.TIMEOUT,
        )
        assert result.is_success is False

    def test_is_success_false_for_error(self) -> None:
        """is_success returns False for ERROR status."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.ERROR,
        )
        assert result.is_success is False

    def test_is_timeout_true(self) -> None:
        """is_timeout returns True for TIMEOUT status."""
        result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.TIMEOUT,
        )
        assert result.is_timeout is True

    def test_is_timeout_false_for_success(self) -> None:
        """is_timeout returns False for SUCCESS status."""
        result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
        )
        assert result.is_timeout is False

    def test_is_timeout_false_for_error(self) -> None:
        """is_timeout returns False for ERROR status."""
        result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.ERROR,
        )
        assert result.is_timeout is False

    def test_is_error_true(self) -> None:
        """is_error returns True for ERROR status."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.ERROR,
        )
        assert result.is_error is True

    def test_is_error_false_for_success(self) -> None:
        """is_error returns False for SUCCESS status."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
        )
        assert result.is_error is False

    def test_is_error_false_for_timeout(self) -> None:
        """is_error returns False for TIMEOUT status."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.TIMEOUT,
        )
        assert result.is_error is False

    # -------------------------------------------------------------------------
    # Serialization tests
    # -------------------------------------------------------------------------

    def test_to_dict_success(self) -> None:
        """to_dict correctly serializes a success result."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="Analysis result",
            execution_time_ms=250.5,
        )

        d = result.to_dict()

        assert d == {
            "agent_name": "news_agent",
            "status": "success",
            "response": "Analysis result",
            "error_message": None,
            "execution_time_ms": 250.5,
        }

    def test_to_dict_timeout(self) -> None:
        """to_dict correctly serializes a timeout result."""
        result = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.TIMEOUT,
            error_message="Timed out",
            execution_time_ms=60000.0,
        )

        d = result.to_dict()

        assert d["agent_name"] == "sec_agent"
        assert d["status"] == "timeout"
        assert d["response"] is None
        assert d["error_message"] == "Timed out"
        assert d["execution_time_ms"] == 60000.0

    def test_to_dict_error(self) -> None:
        """to_dict correctly serializes an error result."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.ERROR,
            error_message="API failure",
            execution_time_ms=100.0,
        )

        d = result.to_dict()

        assert d["status"] == "error"
        assert d["error_message"] == "API failure"

    def test_to_dict_returns_new_dict(self) -> None:
        """to_dict returns a new dictionary each time."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
        )

        d1 = result.to_dict()
        d2 = result.to_dict()

        assert d1 is not d2
        assert d1 == d2

    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------

    def test_empty_response(self) -> None:
        """Result can have empty string response."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="",
        )
        assert result.response == ""
        assert result.is_success is True

    def test_very_long_response(self) -> None:
        """Result can handle very long response."""
        long_response = "x" * 100000
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response=long_response,
        )
        assert result.response == long_response
        response_in_dict = result.to_dict()["response"]
        assert isinstance(response_in_dict, str)
        assert len(response_in_dict) == 100000

    def test_special_characters_in_response(self) -> None:
        """Result can handle special characters."""
        response = 'Test with "quotes", <tags>, and unicode: \u2764\ufe0f'
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response=response,
        )
        assert result.response == response

    def test_zero_execution_time(self) -> None:
        """Result can have zero execution time."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            execution_time_ms=0.0,
        )
        assert result.execution_time_ms == 0.0

    def test_negative_execution_time(self) -> None:
        """Result accepts negative execution time (no validation)."""
        result = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            execution_time_ms=-10.0,
        )
        assert result.execution_time_ms == -10.0

    def test_equality(self) -> None:
        """Two results with same values are equal."""
        result1 = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="Test",
            execution_time_ms=100.0,
        )
        result2 = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
            response="Test",
            execution_time_ms=100.0,
        )
        assert result1 == result2

    def test_inequality(self) -> None:
        """Two results with different values are not equal."""
        result1 = AgentResult(
            agent_name="news_agent",
            status=AgentStatus.SUCCESS,
        )
        result2 = AgentResult(
            agent_name="sec_agent",
            status=AgentStatus.SUCCESS,
        )
        assert result1 != result2


class TestQueryClassification:
    """Tests for QueryClassification dataclass."""

    # -------------------------------------------------------------------------
    # Creation tests
    # -------------------------------------------------------------------------

    def test_create_news_only(self) -> None:
        """Can create a news-only classification."""
        classification = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            confidence=0.95,
            matched_patterns=["news", "sentiment"],
            reasoning="Query mentions news and sentiment",
        )

        assert classification.route_type == RouteType.NEWS_ONLY
        assert classification.confidence == 0.95
        assert classification.matched_patterns == ["news", "sentiment"]
        assert classification.reasoning == "Query mentions news and sentiment"

    def test_create_sec_only(self) -> None:
        """Can create a SEC-only classification."""
        classification = QueryClassification(
            route_type=RouteType.SEC_ONLY,
            confidence=0.85,
            matched_patterns=["10-K", "filing"],
        )

        assert classification.route_type == RouteType.SEC_ONLY
        assert classification.confidence == 0.85

    def test_create_both(self) -> None:
        """Can create a both-agents classification."""
        classification = QueryClassification(
            route_type=RouteType.BOTH,
            confidence=0.9,
            matched_patterns=["compare", "news", "SEC"],
        )

        assert classification.route_type == RouteType.BOTH

    def test_default_values(self) -> None:
        """Default values are applied correctly."""
        classification = QueryClassification(route_type=RouteType.BOTH)

        assert classification.confidence == 1.0
        assert classification.matched_patterns == []
        assert classification.reasoning is None

    def test_matched_patterns_is_mutable(self) -> None:
        """Matched patterns list is mutable."""
        classification = QueryClassification(route_type=RouteType.NEWS_ONLY)
        classification.matched_patterns.append("new_pattern")
        assert "new_pattern" in classification.matched_patterns

    def test_default_list_is_not_shared(self) -> None:
        """Each instance gets its own default list."""
        c1 = QueryClassification(route_type=RouteType.NEWS_ONLY)
        c2 = QueryClassification(route_type=RouteType.SEC_ONLY)

        c1.matched_patterns.append("test")

        assert "test" in c1.matched_patterns
        assert "test" not in c2.matched_patterns

    # -------------------------------------------------------------------------
    # Property tests - needs_news_agent
    # -------------------------------------------------------------------------

    def test_needs_news_agent_for_news_only(self) -> None:
        """needs_news_agent returns True for NEWS_ONLY."""
        classification = QueryClassification(route_type=RouteType.NEWS_ONLY)
        assert classification.needs_news_agent is True

    def test_needs_news_agent_for_both(self) -> None:
        """needs_news_agent returns True for BOTH."""
        classification = QueryClassification(route_type=RouteType.BOTH)
        assert classification.needs_news_agent is True

    def test_needs_news_agent_for_sec_only(self) -> None:
        """needs_news_agent returns False for SEC_ONLY."""
        classification = QueryClassification(route_type=RouteType.SEC_ONLY)
        assert classification.needs_news_agent is False

    # -------------------------------------------------------------------------
    # Property tests - needs_sec_agent
    # -------------------------------------------------------------------------

    def test_needs_sec_agent_for_sec_only(self) -> None:
        """needs_sec_agent returns True for SEC_ONLY."""
        classification = QueryClassification(route_type=RouteType.SEC_ONLY)
        assert classification.needs_sec_agent is True

    def test_needs_sec_agent_for_both(self) -> None:
        """needs_sec_agent returns True for BOTH."""
        classification = QueryClassification(route_type=RouteType.BOTH)
        assert classification.needs_sec_agent is True

    def test_needs_sec_agent_for_news_only(self) -> None:
        """needs_sec_agent returns False for NEWS_ONLY."""
        classification = QueryClassification(route_type=RouteType.NEWS_ONLY)
        assert classification.needs_sec_agent is False

    # -------------------------------------------------------------------------
    # Property tests - needs_both
    # -------------------------------------------------------------------------

    def test_needs_both_for_both_route(self) -> None:
        """needs_both returns True for BOTH route type."""
        classification = QueryClassification(route_type=RouteType.BOTH)
        assert classification.needs_both is True

    def test_needs_both_for_news_only(self) -> None:
        """needs_both returns False for NEWS_ONLY."""
        classification = QueryClassification(route_type=RouteType.NEWS_ONLY)
        assert classification.needs_both is False

    def test_needs_both_for_sec_only(self) -> None:
        """needs_both returns False for SEC_ONLY."""
        classification = QueryClassification(route_type=RouteType.SEC_ONLY)
        assert classification.needs_both is False

    # -------------------------------------------------------------------------
    # Confidence tests
    # -------------------------------------------------------------------------

    def test_confidence_zero(self) -> None:
        """Can have zero confidence."""
        classification = QueryClassification(
            route_type=RouteType.BOTH,
            confidence=0.0,
        )
        assert classification.confidence == 0.0

    def test_confidence_one(self) -> None:
        """Can have full confidence."""
        classification = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            confidence=1.0,
        )
        assert classification.confidence == 1.0

    def test_confidence_above_one(self) -> None:
        """Confidence above 1.0 is accepted (no validation)."""
        classification = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            confidence=1.5,
        )
        assert classification.confidence == 1.5

    def test_confidence_negative(self) -> None:
        """Negative confidence is accepted (no validation)."""
        classification = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            confidence=-0.5,
        )
        assert classification.confidence == -0.5

    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------

    def test_empty_matched_patterns(self) -> None:
        """Classification can have empty matched patterns."""
        classification = QueryClassification(
            route_type=RouteType.BOTH,
            matched_patterns=[],
        )
        assert classification.matched_patterns == []

    def test_many_matched_patterns(self) -> None:
        """Classification can have many matched patterns."""
        patterns = [f"pattern_{i}" for i in range(100)]
        classification = QueryClassification(
            route_type=RouteType.BOTH,
            matched_patterns=patterns,
        )
        assert len(classification.matched_patterns) == 100

    def test_empty_reasoning(self) -> None:
        """Classification can have empty string reasoning."""
        classification = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            reasoning="",
        )
        assert classification.reasoning == ""

    def test_long_reasoning(self) -> None:
        """Classification can have long reasoning."""
        reasoning = "This is a detailed explanation. " * 100
        classification = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            reasoning=reasoning,
        )
        assert classification.reasoning == reasoning

    def test_equality(self) -> None:
        """Two classifications with same values are equal."""
        c1 = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            confidence=0.9,
            matched_patterns=["news"],
            reasoning="Test",
        )
        c2 = QueryClassification(
            route_type=RouteType.NEWS_ONLY,
            confidence=0.9,
            matched_patterns=["news"],
            reasoning="Test",
        )
        assert c1 == c2

    def test_inequality_route_type(self) -> None:
        """Classifications with different route types are not equal."""
        c1 = QueryClassification(route_type=RouteType.NEWS_ONLY)
        c2 = QueryClassification(route_type=RouteType.SEC_ONLY)
        assert c1 != c2

    def test_inequality_confidence(self) -> None:
        """Classifications with different confidence are not equal."""
        c1 = QueryClassification(route_type=RouteType.BOTH, confidence=0.9)
        c2 = QueryClassification(route_type=RouteType.BOTH, confidence=0.8)
        assert c1 != c2


class TestPropertyCombinations:
    """Test property combinations for QueryClassification."""

    def test_news_only_agent_properties(self) -> None:
        """NEWS_ONLY has correct agent requirement properties."""
        classification = QueryClassification(route_type=RouteType.NEWS_ONLY)
        assert classification.needs_news_agent is True
        assert classification.needs_sec_agent is False
        assert classification.needs_both is False

    def test_sec_only_agent_properties(self) -> None:
        """SEC_ONLY has correct agent requirement properties."""
        classification = QueryClassification(route_type=RouteType.SEC_ONLY)
        assert classification.needs_news_agent is False
        assert classification.needs_sec_agent is True
        assert classification.needs_both is False

    def test_both_agent_properties(self) -> None:
        """BOTH has correct agent requirement properties."""
        classification = QueryClassification(route_type=RouteType.BOTH)
        assert classification.needs_news_agent is True
        assert classification.needs_sec_agent is True
        assert classification.needs_both is True


class TestAgentResultStatusCombinations:
    """Test status combination scenarios for AgentResult."""

    @pytest.mark.parametrize(
        "status,expected_success,expected_timeout,expected_error",
        [
            (AgentStatus.SUCCESS, True, False, False),
            (AgentStatus.TIMEOUT, False, True, False),
            (AgentStatus.ERROR, False, False, True),
        ],
    )
    def test_status_properties(
        self,
        status: AgentStatus,
        expected_success: bool,
        expected_timeout: bool,
        expected_error: bool,
    ) -> None:
        """Test that exactly one status property is True for each status."""
        result = AgentResult(agent_name="news_agent", status=status)

        assert result.is_success is expected_success
        assert result.is_timeout is expected_timeout
        assert result.is_error is expected_error

    @pytest.mark.parametrize("status", list(AgentStatus))
    def test_exactly_one_status_property_true(self, status: AgentStatus) -> None:
        """Ensure exactly one status property is True for any status."""
        result = AgentResult(agent_name="news_agent", status=status)

        status_flags = [result.is_success, result.is_timeout, result.is_error]
        assert sum(status_flags) == 1


class TestDataclassHashability:
    """Test hashability properties of dataclasses."""

    def test_agent_result_not_hashable_by_default(self) -> None:
        """AgentResult is not hashable (mutable dataclass)."""
        result = AgentResult(agent_name="news_agent", status=AgentStatus.SUCCESS)
        with pytest.raises(TypeError):
            hash(result)

    def test_query_classification_not_hashable_by_default(self) -> None:
        """QueryClassification is not hashable (has mutable list field)."""
        classification = QueryClassification(route_type=RouteType.NEWS_ONLY)
        with pytest.raises(TypeError):
            hash(classification)
