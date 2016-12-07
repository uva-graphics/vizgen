
import z3_util
import util

def test_prove_smallest():
    assert z3_util.prove_smallest(['1+a', '1', '2'], ['a'], ['a>=1']) == '1'
    assert z3_util.prove_smallest(['1+a', '1', '2'], ['a'], ['a>=0']) == '1'
    assert z3_util.prove_smallest(['1+a', '1', '2'], ['a'], ['a>=-1']) == '1+a'
    util.print_twocol('z3_util.prove_smallest:', 'OK')

if __name__ == '__main__':
    test_prove_smallest()
