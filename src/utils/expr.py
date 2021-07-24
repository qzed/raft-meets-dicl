import ast
import operator as op


def eval_math_expr(expr, args):
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_node(node.operand))
        else:
            raise TypeError(node)

    # substitute variables
    expr = expr.format_map(args)

    # parse syntax tree
    tree = ast.parse(expr, mode='eval')

    # evaluate
    return eval_node(tree.body)
