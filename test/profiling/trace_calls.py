import pathlib, fdp, os, sys
import numpy as np

fdp_module_path = pathlib.Path(fdp.__path__[0])
mds_path = pathlib.Path(os.environ['MDSPLUS']) / 'mdsobjects' / 'python'
python_path = pathlib.Path(sys.base_prefix) / 'lib'
np_path = pathlib.Path(np.__path__[0])

def shorten_filename(fn):
    try:
        fn = fn.relative_to(fdp_module_path)
    except:
        try:
            fn = fn.relative_to(mds_path)
        except:
            try:
                fn = fn.relative_to(python_path)
            except:
                try:
                    fn = fn.relative_to(np_path)
                except:
                    pass
    return fn

def trace_lines(frame, event, arg):
    if event not in ['return']:
        return
    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = pathlib.Path(co.co_filename)
    filename = shorten_filename(filename)
    # if event == 'line':
    #     print('   Executing line  {}  in  {}  in  {}'.format(line_no, func_name, filename))
    if func_name[0] == '<':
        return
    if event == 'return':
        print('   Returning from  {}  in  {}'.format(func_name, filename))


def trace_calls(frame, event, arg):
    if event != 'call':
        # return for non-call events
        print('skipped event: {}'.format(event))
        return
    try:
        # caller details
        caller = frame.f_back
        caller_lineno = caller.f_lineno
        caller_code = caller.f_code
        caller_name = caller_code.co_name
        caller_filename = pathlib.Path(caller_code.co_filename)
        # callee details
        call_code = frame.f_code
        call_name = call_code.co_name
        call_filename = pathlib.Path(call_code.co_filename)
    except Exception as e:
        # print(e.args)
        return
    if call_name == 'write':
        # Ignore write calls from print statements
        return
    if (fdp_module_path.as_posix() not in call_filename.as_posix()) or \
        (fdp_module_path.as_posix() not in caller_filename.as_posix()):
        # ignore calls fully outside of the FDP package
        if caller_name == '<module>':
            print('Line {} in {}'.format(caller_lineno, caller_filename.name))
        else:
            return
    # adjust filenames relative to FDP directory
    caller_shortfn = shorten_filename(caller_filename)
    call_shortfn = shorten_filename(call_filename)
    if call_name[0] == '<':
        # 'parse' in call_name or \
        # 'parse' in caller_name or \
        # '__get' in call_name or \
        # '__get' in caller_name:
        return
    if 'MDSplus' in call_filename.as_posix():
        return
    print('   Calling  {}  in  {}  from line  {}  in  {}  in  {}'.format(
            call_name,
            call_shortfn.as_posix(),
            caller_lineno,
            caller_name,
            caller_shortfn.as_posix()))
    return trace_lines
