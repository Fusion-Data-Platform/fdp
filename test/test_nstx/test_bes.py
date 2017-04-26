

def test_bes(setup_nstx):
    nstx = setup_nstx
    dir(nstx.s141000.bes)
    for signal in nstx.s141000.bes:
        pass
    nstx.s141000.bes.ch01[:]
    nstx.s204320.bes.ch01[:]
