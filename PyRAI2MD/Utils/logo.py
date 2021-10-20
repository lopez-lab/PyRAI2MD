######################################################
#
# PyRAI2MD 2 module for printing logo
#
# Author Jingbai Li
# Sep 29 2021
#
######################################################

def Logo(version):

    credits="""
  --------------------------------------------------------------
                              /\\
   |\\    /|                  /++\\
   ||\\  /||                 /++++\\
   || \\/ || ||             /++++++\\
   ||    || ||            /PyRAI2MD\\
   ||    || ||           /++++++++++\\                    __
            ||          /++++++++++++\\    |\\ |  /\\  |\\/| | \\
            ||__ __    *==============*   | \\| /--\\ |  | |_/

                          Python Rapid
                     Artificial Intelligence
                  Ab Initio Molecular Dynamics



                      Author @Jingbai Li
               Northeastern University, Boston, USA

                          version:   %s


  With contriutions from (in alphabetic order):
    Jingbai Li                 - Fewest switches surface hopping
                                 Zhu-Nakamura surface hopping
                                 Velocity Verlet
                                 OpenMolcas interface
                                 OpenMolcas/Tinker interface
                                 BAGEL interface
                                 Adaptive sampling
                                 Grid search
                                 Two-layer ONIOM (coming soon)
                                 Periodic boundary condition (coming soon)
                                 QC/ML hybrid NAMD

    Patrick Reiser             - Neural networks (pyNNsMD)

  Special acknowledgement to:
    Steven A. Lopez            - Project directorship
    Pascal Friederich          - ML directoriship

""" % (version)

    return credits
