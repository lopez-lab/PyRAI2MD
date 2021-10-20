######################################################
#
# PyRAI2MD 2 module for utility tools - timing
#
# Author Jingbai Li
# Sep 8 2021
#
######################################################

import time, datetime

def WhatIsTime():
    ## This function return current time

    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

def HowLong(start, end):
    ## This function calculate time between start and end

    walltime = end-start
    walltime = '%5d days %5d hours %5d minutes %5d seconds' % (
        int(walltime / 86400),
        int((walltime % 86400) / 3600),
        int(((walltime % 86400) % 3600) / 60),
        int(((walltime % 86400) % 3600) % 60))
    return walltime
