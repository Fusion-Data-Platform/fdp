

def test_bes(setup_nstx):
    nstx = setup_nstx
    dir(nstx.s142301.bes)
    for signal in nstx.s142301.bes:
        assert bool(signal.mdsshape)
    nstx.s142301.bes.ch01[:]
    nstx.s204320.bes.ch01[:]
