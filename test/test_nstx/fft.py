
def test_fft(setup_shot):
    shot = setup_shot
    kwargs = {'tmin': 0.2, 'tmax': 0.3}
    shot.bes.ch01.plotfft(**kwargs)
    shot.bes.ch01.powerspectrum(tmin=0.270, tmax=0.280, fmax=200, power2=4e3)
    shot.magnetics.highn.highn_7.plotfft(**kwargs)
    shot.usxr.hdown.hdown08.plotfft(**kwargs)
