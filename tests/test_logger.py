"""Tests for structured logging module."""

import logging

import pytest

from star_graph.logger import (
    init_logging,
    get_logger,
    shutdown,
    StructuredLog,
)


class TestInitLogging:
    def test_initializes_root_logger(self):
        init_logging(level=logging.INFO)
        root = logging.getLogger("star_graph")
        assert root.level == logging.INFO
        assert len(root.handlers) >= 1  # console handler

    def test_custom_level(self):
        init_logging(level=logging.DEBUG)
        root = logging.getLogger("star_graph")
        assert root.level == logging.DEBUG

    def test_clears_previous_handlers(self):
        init_logging()
        before = len(logging.getLogger("star_graph").handlers)
        init_logging()
        after = len(logging.getLogger("star_graph").handlers)
        assert after > 0
        # Handlers are replaced, not doubled
        assert after <= before + 1


class TestGetLogger:
    def test_returns_logger_with_prefix(self):
        init_logging()
        logger = get_logger("star_graph.sleep")
        assert logger.name == "star_graph.sleep"

    def test_adds_prefix_for_non_star_graph_names(self):
        init_logging()
        logger = get_logger("myapp.mymodule")
        assert logger.name == "star_graph.myapp.mymodule"

    def test_caches_loggers(self):
        init_logging()
        a = get_logger("star_graph.test")
        b = get_logger("star_graph.test")
        assert a is b


class TestShutdown:
    def test_clears_handlers(self):
        init_logging()
        shutdown()
        root = logging.getLogger("star_graph")
        assert len(root.handlers) == 0


class TestStructuredLog:
    class TestClass(StructuredLog):
        pass

    def test_creates_logger(self):
        obj = self.TestClass()
        assert obj._logger.name == "star_graph.TestClass"

    def test_log_info(self, caplog):
        obj = self.TestClass()
        with caplog.at_level(logging.INFO):
            obj.log_info("test message", key="value")
        assert "test message" in caplog.text

    def test_log_warning(self, caplog):
        obj = self.TestClass()
        with caplog.at_level(logging.WARNING):
            obj.log_warning("warning message")
        assert "warning message" in caplog.text

    def test_log_error(self, caplog):
        obj = self.TestClass()
        with caplog.at_level(logging.ERROR):
            obj.log_error("error message")
        assert "error message" in caplog.text
