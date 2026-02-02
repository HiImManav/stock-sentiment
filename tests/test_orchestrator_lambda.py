"""Tests for the Orchestration Agent Lambda handler."""

from __future__ import annotations


class TestMangumHandler:
    """Tests for the Mangum Lambda handler."""

    def test_handler_is_mangum_instance(self) -> None:
        """Handler is a Mangum instance wrapping the FastAPI app."""
        from mangum import Mangum
        from orchestrator.mangum_handler import handler

        assert isinstance(handler, Mangum)

    def test_handler_has_lifespan_off(self) -> None:
        """Handler is configured with lifespan='off'."""
        from orchestrator.mangum_handler import handler

        # Mangum stores the lifespan setting
        assert handler.lifespan == "off"

    def test_handler_app_is_fastapi(self) -> None:
        """Handler wraps a FastAPI application."""
        from fastapi import FastAPI
        from orchestrator.mangum_handler import handler

        assert isinstance(handler.app, FastAPI)

    def test_handler_app_has_correct_title(self) -> None:
        """Handler wraps the orchestrator app with correct title."""
        from orchestrator.mangum_handler import handler

        assert handler.app.title == "Orchestration Agent API"

    def test_handler_app_has_health_route(self) -> None:
        """Handler app has the /health route configured."""
        from orchestrator.mangum_handler import handler

        routes = [route.path for route in handler.app.routes]
        assert "/health" in routes

    def test_handler_app_has_query_route(self) -> None:
        """Handler app has the /query route configured."""
        from orchestrator.mangum_handler import handler

        routes = [route.path for route in handler.app.routes]
        assert "/query" in routes

    def test_handler_app_has_compare_route(self) -> None:
        """Handler app has the /compare route configured."""
        from orchestrator.mangum_handler import handler

        routes = [route.path for route in handler.app.routes]
        assert "/compare" in routes

    def test_handler_app_has_session_summary_route(self) -> None:
        """Handler app has the /session/summary route configured."""
        from orchestrator.mangum_handler import handler

        routes = [route.path for route in handler.app.routes]
        assert "/session/summary" in routes

    def test_handler_app_has_session_history_route(self) -> None:
        """Handler app has the /session/history route configured."""
        from orchestrator.mangum_handler import handler

        routes = [route.path for route in handler.app.routes]
        assert "/session/history" in routes

    def test_handler_app_has_reset_route(self) -> None:
        """Handler app has the /reset route configured."""
        from orchestrator.mangum_handler import handler

        routes = [route.path for route in handler.app.routes]
        assert "/reset" in routes


class TestLambdaInvocation:
    """Tests for Lambda event handling via TestClient simulation."""

    def test_health_check_works(self) -> None:
        """Health check endpoint works via the app wrapped by handler."""
        from fastapi.testclient import TestClient
        from orchestrator.mangum_handler import handler

        # Test via TestClient on the wrapped app
        client = TestClient(handler.app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "orchestrator"


class TestHandlerCallable:
    """Tests that the handler is callable (the Lambda entry point)."""

    def test_handler_is_callable(self) -> None:
        """Handler is callable as required for Lambda."""
        from orchestrator.mangum_handler import handler

        assert callable(handler)

    def test_handler_matches_news_agent_pattern(self) -> None:
        """Handler follows the same pattern as news_agent handler."""
        from mangum import Mangum
        from orchestrator.mangum_handler import handler as orchestrator_handler
        from src.news_agent.mangum_handler import handler as news_handler

        # Both should be Mangum instances with lifespan off
        assert isinstance(orchestrator_handler, Mangum)
        assert isinstance(news_handler, Mangum)
        assert orchestrator_handler.lifespan == news_handler.lifespan == "off"

    def test_handler_matches_sec_agent_pattern(self) -> None:
        """Handler follows the same pattern as sec_agent handler."""
        from mangum import Mangum
        from orchestrator.mangum_handler import handler as orchestrator_handler
        from sec_agent.mangum_handler import handler as sec_handler

        # Both should be Mangum instances with lifespan off
        assert isinstance(orchestrator_handler, Mangum)
        assert isinstance(sec_handler, Mangum)
        assert orchestrator_handler.lifespan == sec_handler.lifespan == "off"
