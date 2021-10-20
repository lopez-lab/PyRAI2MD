
######################################################
#
# PyRAI2MD test first run
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

import os

from TEST.first_run.code_info import register, review

def FirstRun():

    """ first run test

    1. check code completeness
    2. review code structure

    """
    code = 'PASSED'
    pyrai2mddir = os.environ['PYRAI2MD']
    totline = 0
    totfile = 0
    length = {}
    summary = """
 *---------------------------------------------------*
 |                                                   |
 |             Check Code Completeness               |
 |                                                   |
 *---------------------------------------------------*

"""
    for name, location in register.items():
        mod = '%s/%s' % (pyrai2mddir, location)
        status = os.path.exists(mod)
        if status == True:
            totfile += 1
            with open(mod, 'r') as file:
                n = len(file.readlines())
            totline += n
            length[name] = n
            mark = 'Found:'
        else:
            length[name] = 0
            mark = 'Missing:'
            code = 'FAILED(incomplete code)'
        summary += '%-10s %s\n' % (mark, mod)
    summary += """
 *---------------------------------------------------*
 |                                                   |
 |                 Code Structure                    |
 |                                                   |
 *---------------------------------------------------*
"""
    summary += review(length, totline, totfile)

    return summary, code
