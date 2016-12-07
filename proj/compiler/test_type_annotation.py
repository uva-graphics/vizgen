import type_infer
import compiler
import util

def test_type_annotations2():
    s = open('test_programs/type_annotation2.py', 'rt').read()
    
    program_info = compiler.ProgramInfo(s, preallocate_verbose=False)
    typeinfer_d = type_infer.type_infer(program_info)
    
    assert(typeinfer_d['annotated_type']['f'][0]['a'] == typeinfer_d['types']['f'][0]['A'])
    assert(typeinfer_d['annotated_type']['f'][0]['c'] == typeinfer_d['types']['f'][0]['C'])
    
    assert(typeinfer_d['types']['f'][0]['a'] == typeinfer_d['types']['f'][0]['A'])
    assert(typeinfer_d['types']['f'][0]['c'] == typeinfer_d['types']['f'][0]['C'])
    assert(typeinfer_d['types']['f'][0]['d'] == typeinfer_d['types']['f'][0]['D'])
    
    util.print_twocol('test_type_annotations2:', 'OK')

def test_type_annotations():
    import compiler
    s = open('test_programs/type_annotation.py', 'rt').read()
    program_info = compiler.ProgramInfo(s, preallocate=False)
    type_infer_d = type_infer.type_infer(program_info, verbose=False, get_macros=True)
    main_types = type_infer_d['types']['main'][0]
    for varname in ['A', 'C1', 'C2', 'C4', 'C4b']:
        assert(main_types[varname].is_array() and main_types[varname].shape == (2, 2))
    assert(main_types['C3'].is_array() and main_types['C3'].shape == (None, None))
    for varname in ['a', 'b']:
        assert(str(main_types[varname]) == "'int'")

    test_types = type_infer_d['types']['test'][0]
    for varname in ['A', 'B']:
        assert(test_types[varname].is_array() and test_types[varname].shape == (2, 2))
    util.print_twocol('test_type_annotations:', 'OK')

if __name__ == '__main__':
    test_type_annotations()
    test_type_annotations2()
