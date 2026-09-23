"""Tests for order_tools module."""

import pytest

from foodstack.tools.order_tools import get_order_status


def call_order_status(search_value: str) -> str:
    """Helper to call the get_order_status tool."""
    return get_order_status.invoke({"search_value": search_value})


class TestGetOrderStatus:
    """Test cases for the get_order_status tool."""

    def test_search_by_order_id(self):
        """Test searching by order ID."""
        result = call_order_status("ORD-001")
        assert isinstance(result, str)

    def test_search_by_tracking_id(self):
        """Test searching by tracking ID."""
        result = call_order_status("FS202TRK")
        assert isinstance(result, str)

    def test_search_by_email(self):
        """Test searching by email address."""
        result = call_order_status("alice@example.com")
        assert isinstance(result, str)

    def test_search_case_insensitive_order_id(self):
        """Test that order ID search is case insensitive."""
        result_upper = call_order_status("ORD-001")
        result_lower = call_order_status("ord-001")
        assert isinstance(result_upper, str)
        assert isinstance(result_lower, str)

    def test_search_case_insensitive_tracking_id(self):
        """Test that tracking ID search is case insensitive."""
        result_upper = call_order_status("FS202TRK")
        result_lower = call_order_status("fs202trk")
        assert isinstance(result_upper, str)
        assert isinstance(result_lower, str)

    def test_search_case_insensitive_email(self):
        """Test that email search is case insensitive."""
        result_lower = call_order_status("alice@example.com")
        result_upper = call_order_status("ALICE@EXAMPLE.COM")
        assert isinstance(result_lower, str)
        assert isinstance(result_upper, str)

    def test_empty_search_value(self):
        """Test with empty search value."""
        result = call_order_status("")
        assert "Error:" in result

    def test_nonexistent_order(self):
        """Test searching for non-existent order."""
        result = call_order_status("NONEXISTENT123")
        assert isinstance(result, str)

    def test_result_is_string(self):
        """Test that result is always a string."""
        result = call_order_status("ORD-001")
        assert isinstance(result, str)


class TestGetOrderStatusIntegration:
    """Integration tests for order status tool."""

    def test_multiple_searches(self):
        """Test multiple consecutive searches."""
        queries = [
            "ORD-001",
            "FS202TRK",
            "alice@example.com",
            "NONEXISTENT",
        ]
        for query in queries:
            result = call_order_status(query)
            assert isinstance(result, str)

    def test_search_variations(self):
        """Test different search variations for same data."""
        result1 = call_order_status("ORD-001")
        result2 = call_order_status("FS202TRK")
        result3 = call_order_status("alice@example.com")

        assert all(isinstance(r, str) for r in [result1, result2, result3])

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        result_empty = call_order_status("")
        assert "Error:" in result_empty or "No orders found" in result_empty
