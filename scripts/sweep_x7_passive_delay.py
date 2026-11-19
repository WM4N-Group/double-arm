#!/usr/bin/env python3
"""Development-only intermediate engagement delays; same rig and force."""
import audit_x7_passive_brake as brake

DELAYS={f'lost-delay{ms}ms':(ticks,brake.STRONG) for ms,ticks in ((25,6),(50,12),(75,18))}


if __name__=='__main__':
    brake.CASES=DELAYS
    brake.main()
