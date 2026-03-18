"""AST-based lowering of Python functions to KernelSpec for @kernel.elementwise."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ironkernel._facade import KernelFacade
    from ironkernel._ironkernel import Expr, KernelSpec


def lower_function_to_spec(fn: object, kf: KernelFacade) -> KernelSpec:
    """Lower a Python function to a KernelSpec via AST analysis."""
    if not callable(fn):
        msg = "elementwise decorator expects a callable"
        raise TypeError(msg)

    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)

    func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not func_defs:
        msg = "could not find function definition"
        raise SyntaxError(msg)
    func_def = func_defs[0]

    _validate_function(func_def)

    param_names = [a.arg for a in func_def.args.args]
    arg_exprs: dict[str, Any] = {name: kf.arg(name) for name in param_names}

    return_node = _get_single_return(func_def)
    if return_node.value is None:
        msg = "ironkernel.elementwise requires a return value"
        raise SyntaxError(msg)
    expr: Expr = _lower_expr(return_node.value, arg_exprs, kf)

    return kf.elementwise(expr)


def _validate_function(func_def: ast.FunctionDef) -> None:
    """Validate that the function uses only supported AST constructs."""
    args = func_def.args

    if args.vararg:
        msg = "ironkernel.elementwise does not support *args"
        raise SyntaxError(msg)
    if args.kwarg:
        msg = "ironkernel.elementwise does not support **kwargs"
        raise SyntaxError(msg)
    if args.defaults or args.kw_defaults:
        msg = "ironkernel.elementwise does not support default argument values"
        raise SyntaxError(msg)

    param_names = {a.arg for a in args.args}
    allowed_names = param_names | {"kernel", "abs"}

    for node in ast.walk(func_def):
        if isinstance(node, ast.If):
            msg = "ironkernel.elementwise does not support if/else; use kernel.where()"
            raise SyntaxError(msg)
        if isinstance(node, (ast.For, ast.While)):
            kind = "for" if isinstance(node, ast.For) else "while"
            msg = f"ironkernel.elementwise does not support {kind} loops"
            raise SyntaxError(msg)
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            msg = "ironkernel.elementwise does not support assignment statements"
            raise SyntaxError(msg)
        if isinstance(node, ast.BoolOp):
            msg = "ironkernel.elementwise does not support and/or/not; use kernel.where()"
            raise SyntaxError(msg)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            msg = "ironkernel.elementwise does not support and/or/not; use kernel.where()"
            raise SyntaxError(msg)
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            msg = f"ironkernel.elementwise does not support free variable '{node.id}'"
            raise SyntaxError(msg)

    returns = [n for n in ast.walk(func_def) if isinstance(n, ast.Return)]
    if len(returns) == 0:
        msg = "ironkernel.elementwise requires a return statement"
        raise SyntaxError(msg)
    if len(returns) > 1:
        msg = "ironkernel.elementwise requires a single return statement"
        raise SyntaxError(msg)


def _get_single_return(func_def: ast.FunctionDef) -> ast.Return:
    """Extract the single return statement."""
    returns = [n for n in ast.walk(func_def) if isinstance(n, ast.Return)]
    return returns[0]


_BINOP_MAP: dict[type[ast.operator], str] = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.Div: "__truediv__",
    ast.Pow: "__pow__",
}

_CMPOP_MAP: dict[type[ast.cmpop], str] = {
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
}

_KERNEL_METHODS = frozenset(
    {
        "sqrt",
        "abs",
        "log",
        "exp",
        "log2",
        "log10",
        "floor",
        "ceil",
        "sin",
        "cos",
        "tan",
        "pow",
        "atan2",
        "min",
        "max",
        "where",
    }
)


def _lower_expr(
    node: ast.expr,
    args: dict[str, Any],
    kf: KernelFacade,
) -> Any:
    """Recursively lower an AST expression node to an Expr."""
    if isinstance(node, ast.Name):
        if node.id in args:
            return args[node.id]
        msg = f"ironkernel.elementwise does not support free variable '{node.id}'"
        raise SyntaxError(msg)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        msg = f"unsupported constant type: {type(node.value).__name__}"
        raise SyntaxError(msg)

    if isinstance(node, ast.BinOp):
        left = _lower_expr(node.left, args, kf)
        right = _lower_expr(node.right, args, kf)
        op_type = type(node.op)
        if op_type not in _BINOP_MAP:
            msg = f"unsupported binary operator: {op_type.__name__}"
            raise SyntaxError(msg)
        method_name = _BINOP_MAP[op_type]
        return getattr(left, method_name)(right)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand: Any = _lower_expr(node.operand, args, kf)
        return -operand

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            msg = "ironkernel.elementwise supports only single comparisons"
            raise SyntaxError(msg)
        left = _lower_expr(node.left, args, kf)
        right = _lower_expr(node.comparators[0], args, kf)
        cmp_op_type = type(node.ops[0])
        if cmp_op_type not in _CMPOP_MAP:
            msg = f"unsupported comparison operator: {cmp_op_type.__name__}"
            raise SyntaxError(msg)
        method_name = _CMPOP_MAP[cmp_op_type]
        return getattr(left, method_name)(right)

    if isinstance(node, ast.Call):
        return _lower_call(node, args, kf)

    msg = f"unsupported AST node: {type(node).__name__}"
    raise SyntaxError(msg)


def _lower_call(
    node: ast.Call,
    args: dict[str, Any],
    kf: KernelFacade,
) -> Any:
    """Lower a function call AST node."""
    func = node.func

    # kernel.method() calls.
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "kernel"
        and func.attr in _KERNEL_METHODS
    ):
        lowered_args = [_lower_expr(a, args, kf) for a in node.args]
        return getattr(kf, func.attr)(*lowered_args)

    # Built-in abs().
    if isinstance(func, ast.Name) and func.id == "abs":
        if len(node.args) != 1:
            msg = "abs() takes exactly one argument"
            raise SyntaxError(msg)
        return kf.abs(_lower_expr(node.args[0], args, kf))

    msg = f"unsupported function call: {ast.dump(func)}"
    raise SyntaxError(msg)
