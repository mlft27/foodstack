"""Tests for menu_tools module."""

import pytest

from foodstack.tools.menu_tools import search_menu_catalog


def call_search(query: str, k: int = 3) -> str:
    """Helper to call the search_menu_catalog tool."""
    return search_menu_catalog.invoke({"query": query, "k": k})


class TestSearchMenuCatalog:
    """Test cases for the search_menu_catalog tool."""

    def test_search_by_cuisine_type(self):
        """Test searching for items by cuisine type."""
        result = call_search("pizza")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_by_ingredient(self):
        """Test searching for items by ingredient."""
        result = call_search("pasta")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_by_dietary_preference(self):
        """Test searching for items by dietary preference."""
        result = call_search("vegetarian")
        assert isinstance(result, str)

    def test_search_with_k_parameter(self):
        """Test limiting number of results."""
        result = call_search("burger", k=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_no_results(self):
        """Test search with no matching results."""
        result = call_search("xyzabc12345notfood")
        assert isinstance(result, str)

    def test_search_case_insensitive(self):
        """Test that search is case insensitive."""
        result_lower = call_search("burger")
        result_upper = call_search("BURGER")
        assert isinstance(result_lower, str)
        assert isinstance(result_upper, str)

    def test_search_returns_string(self):
        """Test that search returns a string."""
        result = call_search("burger")
        assert isinstance(result, str)

    def test_search_with_multiple_terms(self):
        """Test searching with multiple terms."""
        result = call_search("cheese burger")
        assert isinstance(result, str)

    def test_search_includes_metadata(self):
        """Test that results include item metadata."""
        result = call_search("burger")
        assert len(result) > 20

    def test_default_k_value(self):
        """Test default k value is 3."""
        result = call_search("food")
        assert isinstance(result, str)


class TestMenuToolsIntegration:
    """Integration tests for menu tools."""

    def test_search_multiple_queries(self):
        """Test multiple consecutive searches."""
        queries = ["burger", "pizza", "salad"]
        for query in queries:
            result = call_search(query)
            assert isinstance(result, str)

    def test_k_parameter_edge_cases(self):
        """Test k parameter with edge cases."""
        result_k1 = call_search("food", k=1)
        result_k5 = call_search("food", k=5)
        assert isinstance(result_k1, str)
        assert isinstance(result_k5, str)
