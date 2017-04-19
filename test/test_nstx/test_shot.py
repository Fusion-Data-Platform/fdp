

def test_shot(setup_shot):
    shot = setup_shot
    dir(shot)
    len(shot)
    repr(shot)
    shot.mpts
    getattr(shot, 'chers')
    shot.logbook()
    shot.get_logbook()
    shot.check_efit()