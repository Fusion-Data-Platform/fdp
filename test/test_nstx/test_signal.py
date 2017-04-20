def test_1d_signals(setup_shot):
    shot = setup_shot
    signals = [shot.bes.ch01,
               shot.magnetics.highn.highn_7,
               shot.usxr.hdown.hdown08,
               shot.equilibria.efit01.ipmeas,
               shot.mpts.ld,
               shot.nbi.total_power,
               shot.engineering.ioh,
               shot.rwm.irwm1]
    for signal in signals:
        signal[:]
        assert signal
        assert hasattr(signal, 'size')
        assert signal.size > 0
        assert hasattr(signal, 'time')
        assert signal.time.size > 0
        signal.plot()
        print(signal[0:9])
        signal.axes
        signal(time=[0.1,0.2])

def test_2d_signals(setup_shot):
    shot = setup_shot
    signals = [shot.mpts.te,
               shot.chers.ti]
    for signal in signals:
        signal[:]
        signal.plot()
        signal[0,0:9]