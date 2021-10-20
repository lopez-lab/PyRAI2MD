# Python Rapid Artificial Intelligence Ab Initio Molecular Dynamics
<pre>

                              /\
   |\    /|                  /++\
   ||\  /||                 /++++\
   || \/ || ||             /++++++\
   ||    || ||            /PyRAI2MD\
   ||    || ||           /++++++++++\                    __
            ||          /++++++++++++\    |\ |  /\  |\/| | \
            ||__ __    *==============*   | \| /--\ |  | |_/

                          Python Rapid
                     Artificial Intelligence
                  Ab Initio Molecular Dynamics



                      Author @Jingbai Li
               Northeastern University, Boston, USA

                          version:   2.0 alpha
                          

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
    Pascal Friederich          - ML directoriship>

</pre>
## Features
 - Machine learning nonadibatic molecular dyanmics (ML-NAMD).
 - Neural network training and grid search.
 - Active learning with ML-NAMD trajectories.
 - Support BAGEL, Molcas for QM, and Molcas/Tinker for QM/MM calculations.
 - Support nonadibatic coupling and spin-orbit coupling (Molcas only)
 
## Prerequisite
 - **Python >=3.7** PyRAI2MD is written and tested in Python 3.7.4. Older version of Python is not tested and might not be working properly.
 - **TensorFlow >=2.2** TensorFlow/Keras API is required to load the trained NN models and predict energy and force.
 - **Cython** PyRAI2MD uses Cython library for efficient surface hopping calculation.
 - **Matplotlib/Numpy** Scientifc graphing and numerical library for plotting training statistic and array manipulation.

## Content
<pre>
 File/Folder Name                                  Description                                      
---------------------------------------------------------------------------------------------------
 pyrai2md.py                                       PyRAI2MD interface                              
 PyRAI2MD                                          source codes folder
  |--variables.py                                  PyRAI2MD input reader                           
  |--method.py                                     PyRAI2MD method manager                         
  |--Molecule                                      atom, molecule, trajectory code folder
  |   |--atom.py                                   atomic properties class                         
  |   |--molecule.py                               molecular properties class                      
  |   |--trajectory.py                             trajectory properties class                     
  |   |--pbc_helper.py                             periodic boundary condition functions           
  |    `-qmmm_helper.py                            qmmm functions                                  
  |
  |--Quantum_Chemistry                             quantum chemicial program interface folder
  |   |--qc_molcas.py                              OpenMolcas interface                            
  |   |--qc_bagel.py                               BAGEL interface                                 
  |    `-qc_molcas_tinker                          OpenMolcas/Tinker interface                     
  |
  |--Machine_Learning                              machine learning library interface folder
  |   |--training_data.py                          training data manager                           
  |   |--model_NN.py                               neural network interface                        
  |   |--hypernn.py                                hyperparameter manager                          
  |   |--permutation.py                            data permutation functions                      
  |   |--adaptive_sampling.py                      adaptive sampling class                         
  |   |--grid_search.py                            grid search class                               
  |   |--remote_train.py                           distribute remote training                      
  |    `-pyNNsMD                                   neural network library                         
  |
  |--Dynamics                                      ab initio molecular dynamics code folder
  |   |--aimd.py                                   molecular dynamics class                        
  |   |--mixaimd.py                                ML-QC hybrid molecular dynamics class           
  |   |--single_point.py                           single point calculation                        
  |   |--hop_probability.py                        surface hopping probability calculation         
  |   |--reset_velocity.py                         velocity adjustment functions                   
  |   |--verlet.py                                 velocity verlet method                          
  |   |--Ensembles                                 thermodynamics control code folder
  |   |   |--ensemble.py                           thermodynamics ensemble manager                 
  |   |   |--microcanonical.py                     microcanonical ensemble                         
  |   |    `-thermostat.py                         canonical ensemble                              
  |   |
  |    `-Propagators                               electronic propagation code folder
  |       |--surface_hopping.py                    surface hopping manager                         
  |       |--fssh.pyx                              fewest switches surface hopping method          
  |       |--gsh.py                                generalized surface hopping method              
  |        `-tsh_helper.py                         trajectory surface hopping tools                
  |
   `-Utils                                         utility folder
      |--aligngeom.py                              geometry aligment and comparison functions      
      |--coordinates.py                            coordinates writing functions                   
      |--read_tools.py                             index reader                                    
      |--bonds.py                                  bond length library                            
      |--sampling.py                               initial condition sampling functions            
      |--timing.py                                 timing functions                                
       `-logo.py                                   logo and credits                                    
</pre>

## Installation
Download the repository

    git clone https://github.com/lopez-lab/PyRAI2MD.git

Specify environment variable of PyRAI2MD

    export PYRAI2MD=/path/to/PyRAI2MD
    
## Test PyRAI2MD
Copy the test script and modify environment variables 

    cp $PYRAI2MD/Tool/test_PyRAI2MD.sh .
    bash test_PyRAI2MD.sh

Or directly run if environment variables are set

    $PYRAI2MD/pyrai2md.py quicktest
    
## Run PyRAI2MD

    $PYRAI2MD/pyrai2md.py input
