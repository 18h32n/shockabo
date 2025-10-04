"""
AST generation utilities for creating clean Python code from DSL operations.

Provides functions for building Python AST nodes from DSL operations,
ensuring proper structure, type safety, and runtime checks.
"""

import ast


class ASTGenerator:
    """Generates Python AST nodes from DSL operations."""

    def __init__(self):
        self.temp_var_counter = 0
        self.operation_counter = 0

    def generate_function_ast(self, function_name: str, body: list[ast.stmt],
                            decorators: list[ast.expr] | None = None) -> ast.FunctionDef:
        """
        Generate a complete function AST.

        Args:
            function_name: Name of the function
            body: List of statements for the function body
            decorators: Optional list of decorator expressions

        Returns:
            ast.FunctionDef node
        """
        args = ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg='grid', annotation=self._make_type_annotation('List[List[int]]'))
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        )

        returns = self._make_type_annotation('List[List[int]]')

        func_def = ast.FunctionDef(
            name=function_name,
            args=args,
            body=body,
            decorator_list=decorators or [],
            returns=returns
        )

        return func_def

    def generate_bounds_check_ast(self, grid_var: str, operation_name: str) -> list[ast.stmt]:
        """
        Generate AST nodes for bounds checking.

        Args:
            grid_var: Name of the grid variable
            operation_name: Name of the operation for error context

        Returns:
            List of AST statements for bounds checking
        """
        checks = []

        # Check if grid is empty
        empty_check = ast.If(
            test=ast.Compare(
                left=ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                             args=[ast.Name(id=grid_var, ctx=ast.Load())], keywords=[]),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value=0)]
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='ExecutionError', ctx=ast.Load()),
                        args=[ast.Constant(value=f"Empty grid provided to {operation_name}")],
                        keywords=[ast.keyword(arg='operation_context', value=ast.Constant(value=operation_name))]
                    )
                )
            ],
            orelse=[]
        )
        checks.append(empty_check)

        # Check grid dimensions
        dim_check = ast.If(
            test=ast.BoolOp(
                op=ast.Or(),
                values=[
                    ast.Compare(
                        left=ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                     args=[ast.Name(id=grid_var, ctx=ast.Load())], keywords=[]),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=30)]
                    ),
                    ast.Compare(
                        left=ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                     args=[ast.Subscript(
                                         value=ast.Name(id=grid_var, ctx=ast.Load()),
                                         slice=ast.Constant(value=0),
                                         ctx=ast.Load()
                                     )], keywords=[]),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=30)]
                    )
                ]
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='ExecutionError', ctx=ast.Load()),
                        args=[ast.Constant(value=f"Grid dimensions exceed 30x30 in {operation_name}")],
                        keywords=[ast.keyword(arg='operation_context', value=ast.Constant(value=operation_name))]
                    )
                )
            ],
            orelse=[]
        )
        checks.append(dim_check)

        return checks

    def generate_type_check_ast(self, value: str, expected_type: str, operation_name: str) -> ast.If:
        """
        Generate AST node for type checking.

        Args:
            value: Name of the value to check
            expected_type: Expected type as string
            operation_name: Operation name for error context

        Returns:
            ast.If node for type checking
        """
        type_check = ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Call(
                    func=ast.Name(id='isinstance', ctx=ast.Load()),
                    args=[
                        ast.Name(id=value, ctx=ast.Load()),
                        self._parse_type_string(expected_type)
                    ],
                    keywords=[]
                )
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id='TypeError', ctx=ast.Load()),
                        args=[ast.Constant(value=f"Invalid type for {value} in {operation_name}")],
                        keywords=[]
                    )
                )
            ],
            orelse=[]
        )

        return type_check

    def generate_numpy_import(self) -> ast.Import:
        """Generate numpy import statement."""
        return ast.Import(names=[ast.alias(name='numpy', asname='np')])

    def generate_typing_imports(self) -> ast.ImportFrom:
        """Generate typing imports."""
        return ast.ImportFrom(
            module='typing',
            names=[
                ast.alias(name='List', asname=None),
                ast.alias(name='Dict', asname=None),
                ast.alias(name='Any', asname=None),
                ast.alias(name='Tuple', asname=None)
            ],
            level=0
        )

    def generate_temp_var_name(self) -> str:
        """Generate a unique temporary variable name."""
        self.temp_var_counter += 1
        return f"_temp_{self.temp_var_counter}"

    def generate_operation_tracking(self, operation_name: str) -> list[ast.stmt]:
        """
        Generate AST nodes for operation performance tracking.

        Args:
            operation_name: Name of the operation to track

        Returns:
            List of AST statements for tracking
        """
        self.operation_counter += 1
        op_id = f"op_{self.operation_counter}"

        # Start timing
        start_time = ast.Assign(
            targets=[ast.Name(id=f'{op_id}_start', ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='time', ctx=ast.Load()),
                    attr='time',
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            )
        )

        return [start_time]

    def _make_type_annotation(self, type_str: str) -> ast.expr:
        """Convert a type string to an AST type annotation."""
        # Simple parser for type annotations like List[List[int]]
        if type_str == 'int':
            return ast.Name(id='int', ctx=ast.Load())
        elif type_str == 'List[List[int]]':
            return ast.Subscript(
                value=ast.Name(id='List', ctx=ast.Load()),
                slice=ast.Subscript(
                    value=ast.Name(id='List', ctx=ast.Load()),
                    slice=ast.Name(id='int', ctx=ast.Load()),
                    ctx=ast.Load()
                ),
                ctx=ast.Load()
            )
        else:
            # Fallback for complex types
            return ast.Name(id='Any', ctx=ast.Load())

    def _parse_type_string(self, type_str: str) -> ast.expr:
        """Parse a type string into an AST expression."""
        if type_str == 'int':
            return ast.Name(id='int', ctx=ast.Load())
        elif type_str == 'list':
            return ast.Name(id='list', ctx=ast.Load())
        elif type_str == 'dict':
            return ast.Name(id='dict', ctx=ast.Load())
        else:
            return ast.Name(id='object', ctx=ast.Load())
