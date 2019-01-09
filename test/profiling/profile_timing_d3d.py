"""
Profile execution time of function calls with cProfile (not line-by-line timing)
"""

import cProfile, pstats
import fdp
import time

pro = cProfile.Profile(builtins=False)


d3d = fdp.D3D() # ~ 4 ms exec time
shot = d3d.s176778 # ~ 1 ms exec time
mag = shot.magnetics # 7 ms exec time
ip = mag.ip
time.sleep(5)
pro.enable()
ip[:]
pro.disable()


ps = pstats.Stats(pro)
ps.sort_stats('cumtime').print_stats(20)
# ps.sort_stats('cumtime').print_stats('numpy', 20)
# ps.sort_stats('cumtime').print_stats('signal', 20)