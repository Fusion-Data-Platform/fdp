
def test_crossphase(setup_nstx):
    nstx = setup_nstx
    bes = nstx.s204990.bes
    bes.plotcrossphase('ch42','ch46',tmin=0.3,tmax=0.4,nperseg=2000,spectrum=False)