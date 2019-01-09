"""
Line-by-line timing of functions/methods with kernprof/line_profiler
"""

import line_profiler
import fdp
import MDSplus as mds
import time
import numpy as np

profile = line_profiler.LineProfiler()

# add function/methods or full modules for profiling
# profile.add_function(fdp.lib.machine.Machine.__init__)
# profile.add_function(fdp.lib.machine.Machine._make_server_connections)
profile.add_function(fdp.lib.machine.Machine._get_mdsdata)
# profile.add_function(fdp.lib.machine.Machine._get_connection)
# profile.add_function(fdp.lib.machine.Machine._set_modules)
# profile.add_function(fdp.lib.machine.Machine.__getattr__)
# profile.add_function(mds.Connection.__init__)
profile.add_function(mds.Connection.get)
# profile.add_function(mds.Connection.closeAllTrees)
# profile.add_function(mds.Connection.openTree)
profile.add_function(mds.Connection.__getAnswer__)
# profile.add_function(fdp.lib.shot.Shot.__init__)
# profile.add_function(fdp.lib.shot.Shot.__getattr__)
# profile.add_function(fdp.lib.container.init_class)
# profile.add_function(fdp.lib.container.container_factory)
# profile.add_function(fdp.lib.container.Container.__init__)
# profile.add_function(fdp.lib.container.Container._set_dynamic_containers)
# profile.add_function(fdp.lib.parse.parse_submachine)
# profile.add_function(fdp.lib.parse.parse_signal)
# profile.add_function(fdp.lib.parse.parse_methods)
# profile.add_function(fdp.lib.parse.parse_mdspath)
# profile.add_function(fdp.lib.signal.Signal.__getitem__)
# profile.add_function(fdp.lib.signal.Signal._get_mdsdata)
# profile.add_function(fdp.lib.signal.Signal.__getattr__)
profile.add_function(fdp.lib.signal.Signal.__array_finalize__)
# profile.add_module(fdp.lib.signal)
# profile.add_function(np.core._internal._ctypes.__init__)
# profile.add_function(np.core._internal._ctypes.get_data)


d3d = fdp.D3D()
shot = d3d.s176778
mag = shot.magnetics # 50 ms exec time
ip = mag.ip
time.sleep(4)
profile.enable()
ip[:]
profile.disable()

profile.print_stats()