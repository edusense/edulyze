"""
This file contains list of implemented drivers for edulyze, it can be extended as more and more sensing systems are
tested with.
Author: Prasoon Patidar
Created at: 26th Sept, 2022
"""
from .edusense.driver import EdusenseDriver
from .moodoo.driver import MoodooDriver


available_drivers = {
    'edusense': EdusenseDriver,
    'moodoo':MoodooDriver
}


def get_driver(driver_name):
    if driver_name in available_drivers:
        return available_drivers[driver_name]
    else:
        raise Exception("InputDriverNotImplementedException")
