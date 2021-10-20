######################################################
#
# PyRAI2MD 2 module for loading QM and ML method
#
# Author Jingbai Li
# Sep 18 2021
#
######################################################

from PyRAI2MD.Quantum_Chemistry.qc_molcas import MOLCAS
from PyRAI2MD.Quantum_Chemistry.qc_bagel import BAGEL
from PyRAI2MD.Quantum_Chemistry.qc_molcas_tinker import MOLCAS_TINKER
from PyRAI2MD.Machine_Learning.model_NN import DNN

class QM:
    """ Electronic structure method class

        Parameters:          Type:
            qm               str         electronic structure method
            keywords         dict        input keywords
            id               int         calculation ID

        Attribute:           Type:

        Functions:           Returns:
            train            self        train a model if qm == 'nn'
            load             self        load a model if qm == 'nn'
            appendix         self        add more information to the selected method
            evaluate         self        run the selected method

    """

    def __init__(self, qm, keywords = None, id = None):
        qm_list  = {
            'molcas' : MOLCAS,
            'mlctkr' : MOLCAS_TINKER,
            'bagel'  : BAGEL,
            'nn'     : DNN,
        }

        self.method = qm_list[qm](keywords = keywords, id = id) # This should pass hypers

    def train(self):
        metrics = self.method.train()
        return metrics

    def load(self):                    #This should load model
        self.method.load()
        return self

    def appendix(self,addons):         #appendix function to pass more info for different methods
        self.method.appendix(addons)
        return self

    def evaluate(self, traj):
        traj = self.method.evaluate(traj)
        return traj
