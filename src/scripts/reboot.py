"""
Reboot a Crazyflie remotely using a Crazyradio
"""

import sys
import time
from cflib.utils.power_switch import PowerSwitch

URI = [
    "radio://0/80/2M/E7E7E7E7E7",
    "radio://0/80/2M/E7E7E7E701",
    "radio://0/80/2M/E7E7E7E702",
    "radio://0/80/2M/E7E7E7E703",
    "radio://0/80/2M/E7E7E7E704",
    "radio://0/80/2M/E7E7E7E705",
]

for uri in URI:
    try:
        print("Rebooting Crazyflie on URI: %s" % uri)
        PowerSwitch(uri).stm_power_cycle()
    except Exception as e:
        print("Could not reboot Crazyflie on URI: %s" % uri)
        print(e)
    # PowerSwitch(uri).stm_power_cycle()
