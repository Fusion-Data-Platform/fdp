def test_1d_signals(setup_shot):
    shot = setup_shot
    signals = [shot.bes.ch01,
               shot.magnetics.highn.highn_7,
               shot.usxr.hdown.hdown08,
               shot.equilibria.efit01.ipmeas,
               shot.mpts.ld,
               shot.nbi.total_power,
               shot.engineering.ioh,
               shot.rwn.irwm1]
    for signal in signals:
        signal[:]
        signal.plot()
        print(signal[0:9])
        signal.axes