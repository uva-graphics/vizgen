
import astor

class EvalError(Exception):
    pass

def eval_node(node):
    if node is not None:
        try:
            return eval(astor.to_source(node))
        except:
            raise EvalError
    return None
