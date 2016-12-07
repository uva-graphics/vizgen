
from redbaron_util import *
import util

def test_is_int_constant():
    assert is_int_constant(' 1')
    assert is_int_constant('10')
    assert is_int_constant('-1 ')
    assert is_int_constant(' -25')
    assert is_int_constant(' 1L')
    assert is_int_constant(' -30L ')
    assert is_int_constant(' 0xf50 ')
    assert is_int_constant(' 0o577 ')
    assert is_int_constant(' -0xF50 ')
    assert is_int_constant(' -0o571 ')
    assert not is_int_constant(' 1.5 ')
    assert not is_int_constant('3.5')
    assert not is_int_constant('1.5e5 ')
    assert not is_int_constant('1 + 3')
    assert not is_int_constant('1 + a')
    util.print_twocol('redbaron_util.is_int_constant:', 'OK')
    
if __name__ == '__main__':
    test_is_int_constant()
