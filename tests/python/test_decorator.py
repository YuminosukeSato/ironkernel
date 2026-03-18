"""Tests for @kernel.elementwise decorator (Phase 5)."""

import ast

import numpy as np
import numpy.testing as npt
import pytest
from ironkernel import _decorator as decorator_module
from ironkernel import kernel, rt
from ironkernel._decorator import _lower_call, _lower_expr, lower_function_to_spec


def _compile_func(src: str):
    """Compile source text into a function object for testing AST rejection."""
    ns: dict = {}
    exec(compile(src, "<test>", "exec"), ns)
    for v in ns.values():
        if callable(v):
            return v
    msg = "no function found in source"
    raise ValueError(msg)


def _parse_expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


class TestDecoratorBasic:
    def test_saxpy_decorator(self):
        """@kernel.elementwise produces a working SAXPY kernel."""

        @kernel.elementwise
        def saxpy(a, x, y):
            return a * x + y

        buf_a = rt.asarray(np.array([2.0, 2.0, 2.0]))
        buf_x = rt.asarray(np.array([1.0, 2.0, 3.0]))
        buf_y = rt.asarray(np.array([10.0, 20.0, 30.0]))
        result = rt.go(kernel.map(saxpy, a=buf_a, x=buf_x, y=buf_y)).result()
        npt.assert_array_equal(result.numpy(), np.array([12.0, 24.0, 36.0]))

    def test_returns_kernel_spec(self):
        """Decorator returns a KernelSpec object."""
        from ironkernel import KernelSpec

        @kernel.elementwise
        def identity(x):
            return x

        assert isinstance(identity, KernelSpec)

    def test_arg_order_matches_function(self):
        """Argument order in KernelSpec matches the function signature."""

        @kernel.elementwise
        def ordered(a, b, c):
            return a + b + c

        buf_a = rt.asarray(np.array([1.0]))
        buf_b = rt.asarray(np.array([10.0]))
        buf_c = rt.asarray(np.array([100.0]))
        result = rt.go(kernel.map(ordered, a=buf_a, b=buf_b, c=buf_c)).result()
        npt.assert_array_equal(result.numpy(), np.array([111.0]))

    def test_math_functions_supported(self):
        """math.sqrt, abs, and other math functions work in decorator."""

        @kernel.elementwise
        def math_ops(x):
            return kernel.sqrt(kernel.abs(x))

        buf = rt.asarray(np.array([4.0, 9.0, 16.0]))
        result = rt.go(kernel.map(math_ops, x=buf)).result()
        npt.assert_array_equal(result.numpy(), np.array([2.0, 3.0, 4.0]))

    def test_builtin_abs_supported_why_documented_python_shortcuts_should_lower_without_backend_escape_hatches(self):
        @kernel.elementwise
        def magnitude(x):
            return abs(x)

        buf = rt.asarray(np.array([-4.0, 0.0, 9.0]))
        result = rt.go(kernel.map(magnitude, x=buf)).result()
        npt.assert_array_equal(result.numpy(), np.array([4.0, 0.0, 9.0]))

    def test_kernel_where_in_decorator(self):
        """kernel.where() works inside decorated functions."""

        @kernel.elementwise
        def relu(x):
            return kernel.where(x > 0, x, 0)

        buf = rt.asarray(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        result = rt.go(kernel.map(relu, x=buf)).result()
        npt.assert_array_equal(result.numpy(), np.array([0.0, 0.0, 0.0, 1.0, 2.0]))

    def test_comparison_operators_all(self):
        """All 6 comparison operators work in decorator."""

        @kernel.elementwise
        def gt(x):
            return kernel.where(x > 0, 1, 0)

        @kernel.elementwise
        def ge(x):
            return kernel.where(x >= 0, 1, 0)

        @kernel.elementwise
        def lt(x):
            return kernel.where(x < 0, 1, 0)

        @kernel.elementwise
        def le(x):
            return kernel.where(x <= 0, 1, 0)

        @kernel.elementwise
        def eq(x):
            return kernel.where(x == 0, 1, 0)

        @kernel.elementwise
        def ne(x):
            return kernel.where(x != 0, 1, 0)

        buf = rt.asarray(np.array([-1.0, 0.0, 1.0]))
        assert rt.go(kernel.map(gt, x=buf)).result().numpy().tolist() == [0.0, 0.0, 1.0]
        assert rt.go(kernel.map(ge, x=buf)).result().numpy().tolist() == [0.0, 1.0, 1.0]
        assert rt.go(kernel.map(lt, x=buf)).result().numpy().tolist() == [1.0, 0.0, 0.0]
        assert rt.go(kernel.map(le, x=buf)).result().numpy().tolist() == [1.0, 1.0, 0.0]
        assert rt.go(kernel.map(eq, x=buf)).result().numpy().tolist() == [0.0, 1.0, 0.0]
        assert rt.go(kernel.map(ne, x=buf)).result().numpy().tolist() == [1.0, 0.0, 1.0]

    def test_multiple_args_order(self):
        """3+ arguments maintain correct positional binding."""

        @kernel.elementwise
        def weighted_sum(w1, w2, w3, x1, x2, x3):
            return w1 * x1 + w2 * x2 + w3 * x3

        bufs = {
            "w1": rt.asarray(np.array([1.0])),
            "w2": rt.asarray(np.array([2.0])),
            "w3": rt.asarray(np.array([3.0])),
            "x1": rt.asarray(np.array([10.0])),
            "x2": rt.asarray(np.array([20.0])),
            "x3": rt.asarray(np.array([30.0])),
        }
        result = rt.go(kernel.map(weighted_sum, **bufs)).result()
        # 1*10 + 2*20 + 3*30 = 10 + 40 + 90 = 140
        npt.assert_array_equal(result.numpy(), np.array([140.0]))

    def test_unary_neg(self):
        """Unary negation works in decorator."""

        @kernel.elementwise
        def negate(x):
            return -x

        buf = rt.asarray(np.array([1.0, -2.0, 3.0]))
        result = rt.go(kernel.map(negate, x=buf)).result()
        npt.assert_array_equal(result.numpy(), np.array([-1.0, 2.0, -3.0]))

    def test_constant_int_and_float(self):
        """Integer and float constants work in decorator body."""

        @kernel.elementwise
        def with_const(x):
            return x * 2 + 0.5

        buf = rt.asarray(np.array([1.0, 2.0]))
        result = rt.go(kernel.map(with_const, x=buf)).result()
        npt.assert_array_equal(result.numpy(), np.array([2.5, 4.5]))

    def test_power_operator(self):
        """** (power) operator works in decorator."""

        @kernel.elementwise
        def square(x):
            return x**2

        buf = rt.asarray(np.array([2.0, 3.0, 4.0]))
        result = rt.go(kernel.map(square, x=buf)).result()
        npt.assert_array_equal(result.numpy(), np.array([4.0, 9.0, 16.0]))


class TestDecoratorRejections:
    def test_reject_non_callable_why_public_entrypoint_must_fail_fast_for_invalid_decorator_input(self):
        with pytest.raises(TypeError, match="callable"):
            lower_function_to_spec(123, kernel)

    def test_reject_source_without_function_definition_why_lowering_requires_real_function_syntax(self, monkeypatch):
        monkeypatch.setattr(decorator_module.inspect, "getsource", lambda _fn: "value = 1\n")

        with pytest.raises(SyntaxError, match="function definition"):
            lower_function_to_spec(lambda x: x, kernel)

    def test_reject_if_else(self):
        def fn_with_if(x):
            if x > 0:
                return x
            return -x

        with pytest.raises(SyntaxError, match="if/else"):
            kernel.elementwise(fn_with_if)

    def test_reject_loop(self):
        with pytest.raises(SyntaxError, match="for"):

            @kernel.elementwise
            def bad(x):
                for i in range(10):
                    x = x + i
                return x

    def test_reject_assignment(self):
        with pytest.raises(SyntaxError, match="assignment"):

            @kernel.elementwise
            def bad(x):
                y = x + 1
                return y

    def test_reject_star_args(self):
        with pytest.raises(SyntaxError, match="\\*args"):

            @kernel.elementwise
            def bad(*args):
                return args[0]

    def test_reject_kwargs_why_keyword_splat_obscures_supported_argument_contract(self):
        with pytest.raises(SyntaxError, match="\\*\\*kwargs"):

            @kernel.elementwise
            def bad(x, **kwargs):
                return x

    def test_reject_default_args(self):
        with pytest.raises(SyntaxError, match="default"):

            @kernel.elementwise
            def bad(x, y=1):
                return x + y

    def test_reject_and_or_not(self):
        with pytest.raises(SyntaxError, match="and/or/not"):

            @kernel.elementwise
            def bad(x, y):
                return x and y

    def test_reject_not_operator_why_python_truthiness_has_no_expr_level_runtime_meaning(self):
        with pytest.raises(SyntaxError, match="and/or/not"):

            @kernel.elementwise
            def bad(x):
                return not x

    def test_reject_closure_variable(self):
        z = 42
        with pytest.raises(SyntaxError, match="free variable"):

            @kernel.elementwise
            def bad(x):
                return x + z

    def test_reject_multiple_returns(self):
        with pytest.raises(SyntaxError, match="single return"):

            @kernel.elementwise
            def bad(x):
                return x
                return x + 1

    def test_reject_no_return(self):
        with pytest.raises(SyntaxError, match="return"):

            @kernel.elementwise
            def bad(x):
                x + 1

    def test_reject_bare_return_why_elementwise_requires_an_expression_value(self):
        with pytest.raises(SyntaxError, match="return value"):

            @kernel.elementwise
            def bad(x):
                return


class TestDecoratorInternalErrorPaths:
    def test_lower_expr_rejects_free_name_why_unbound_symbols_would_break_runtime_binding(self):
        with pytest.raises(SyntaxError, match="free variable 'missing'"):
            _lower_expr(_parse_expr("missing"), {"x": kernel.arg("x")}, kernel)

    def test_lower_expr_rejects_non_numeric_constant_why_python_objects_cannot_cross_into_the_ir(self):
        with pytest.raises(SyntaxError, match="unsupported constant type: str"):
            _lower_expr(_parse_expr("'bad'"), {}, kernel)

    def test_lower_expr_rejects_unsupported_binary_operator_why_ir_only_supports_numeric_math(self):
        with pytest.raises(SyntaxError, match="unsupported binary operator: BitOr"):
            _lower_expr(_parse_expr("x | y"), {"x": kernel.arg("x"), "y": kernel.arg("y")}, kernel)

    def test_lower_expr_rejects_chained_comparison_why_lowering_requires_a_single_comparison_operator(self):
        with pytest.raises(SyntaxError, match="single comparisons"):
            _lower_expr(
                _parse_expr("x < y < z"),
                {"x": kernel.arg("x"), "y": kernel.arg("y"), "z": kernel.arg("z")},
                kernel,
            )

    def test_lower_expr_rejects_unsupported_comparison_operator_why_identity_checks_have_no_numeric_ir_form(self):
        with pytest.raises(SyntaxError, match="unsupported comparison operator: Is"):
            _lower_expr(_parse_expr("x is y"), {"x": kernel.arg("x"), "y": kernel.arg("y")}, kernel)

    def test_lower_expr_rejects_unsupported_ast_node_why_tuple_literals_do_not_lower_to_exprs(self):
        with pytest.raises(SyntaxError, match="unsupported AST node: Tuple"):
            _lower_expr(_parse_expr("(x, y)"), {"x": kernel.arg("x"), "y": kernel.arg("y")}, kernel)

    def test_lower_call_rejects_abs_wrong_arity_why_builtin_shortcuts_must_match_public_api_shape(self):
        with pytest.raises(SyntaxError, match="abs\\(\\) takes exactly one argument"):
            _lower_call(
                _parse_expr("abs(x, y)"),
                {"x": kernel.arg("x"), "y": kernel.arg("y")},
                kernel,
            )

    def test_lower_call_rejects_unsupported_function_why_only_documented_kernel_calls_should_lower(self):
        with pytest.raises(SyntaxError, match="unsupported function call"):
            _lower_call(_parse_expr("len(x)"), {"x": kernel.arg("x")}, kernel)
