"""
Trace function call sequence with sys.settrace()
"""

import sys
import trace_calls
import fdp

d3d = fdp.D3D()
shot = d3d.s176778
mag = shot.magnetics

sys.settrace(trace_calls.trace_calls)
ip = mag.ip
ip[:]