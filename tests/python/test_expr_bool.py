"""Expr.__bool__ must always raise TypeError.

Prevents a bug where using Expr in Python's bool context (if, and, or, not, bool())
would silently evaluate as truthy.
"""

import pytest
from ironkernel import kernel


class TestExprBoolTrap:
    """Expr.__bool__ must always raise TypeError."""

    def test_bool_of_arg_raises(self):
        x = kernel.arg("x")
        with pytest.raises(TypeError, match="cannot convert Expr to bool"):
            bool(x)

    def test_if_expr_raises(self):
        x = kernel.arg("x")
        with pytest.raises(TypeError, match="cannot convert Expr to bool"):
            bool(x > 0)

    def test_and_expr_raises(self):
        x = kernel.arg("x")
        y = kernel.arg("y")
        with pytest.raises(TypeError, match="cannot convert Expr to bool"):
            _ = x and y

    def test_or_expr_raises(self):
        x = kernel.arg("x")
        y = kernel.arg("y")
        with pytest.raises(TypeError, match="cannot convert Expr to bool"):
            _ = x or y

    def test_not_expr_raises(self):
        x = kernel.arg("x")
        with pytest.raises(TypeError, match="cannot convert Expr to bool"):
            _ = not x

    def test_const_zero_is_not_falsy(self):
        """0.0 as Expr must not silently be falsy."""
        expr = kernel.arg("x") * 0
        with pytest.raises(TypeError, match="cannot convert Expr to bool"):
            bool(expr)
