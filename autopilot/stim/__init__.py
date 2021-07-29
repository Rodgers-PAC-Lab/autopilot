from autopilot import prefs

# Put this ahead of the next import to avoid circularity problems
class Stim(object):
    """
    Placeholder stimulus meta-object until full implementation
    """

from autopilot.stim.managers import Stim_Manager, Proportional, init_manager


if prefs.get('AGENT') == "pilot":
    if 'AUDIO' in prefs.get('CONFIG'):
        from autopilot.stim.sound import sounds


