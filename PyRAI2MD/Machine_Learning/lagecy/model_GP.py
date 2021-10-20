######################################################
#
# PyRAI2MD 2 module for interfacing to gp_pes
#
# Author Jingbai Li
# May 21 2021
#
######################################################

import time,datetime,json
import numpy as np
from gp_pes import GaussianProcessPes 

class GPR:
    ## This is the interface to GP

    def __init__(self,variables_all,id=None):
        ## data      : dict
        ##             All data from traning data
        ## pred_data : str
        ##             Filename for test set
        ## x         : np.array
        ##             Inverse distance in shape of (batch,(atom*atom-1)/2)
        ## y_dict    : dict
        ##             Dictionary of y values for each model. Energy in Bohr, Gradients in Hatree/Bohr. Nac are unchanged.

        title           = variables_all['control']['title']
        variables	= variables_all['gp']
        data            = variables['postdata']
        self.version    = variables_all['version']
        self.pred_data  = variables['pred_data']
        self.model	= GaussianProcessPes()
        if id == None or id == 1:
            self.name   = f"GP-{title}"
        else:
            self.name   = f"GP-{title}-{id}"
        self.ncpu	= np.amin([variables_all['control']['ml_ncpu'],3])
        self.modelfile  = variables['modelfile']
        self.silent     = variables['silent']
        self.natom      = data['natom']
        self.nstate     = data['nstate']
        self.size_train = len(data['energy_train'])
        self.size_val   = len(data['energy_val'])
        self.npair      = data['npair']
        self.x          = (data['invr_train'] -data['mid_invr'])/data['dev_invr']
        self.x_val      = (data['invr_val']   -data['mid_invr'])/data['dev_invr']
        self.y_dict={
        'e'  : (data['energy_train'].reshape([self.size_train,-1])  -data['mid_energy']  )/data['dev_energy'],
        'g'  : (data['gradient_train'].reshape([self.size_train,-1])-data['mid_gradient'])/data['dev_gradient'],
        'n'  : (data['nac_train'].reshape([self.size_train,-1])     -data['mid_nac'])     /data['dev_nac'],
        }
        self.y_val_dict={
        'e'  : (data['energy_val'].reshape([self.size_val,-1])   -data['mid_energy']  )/data['dev_energy'],
        'g'  : (data['gradient_val'].reshape([self.size_val,-1]) -data['mid_gradient'])/data['dev_gradient'],
        'n'  : (data['nac_val'].reshape([self.size_val,-1])      -data['mid_nac'])     /data['dev_nac'],
        }
        self.sgm_list={
        'invr': data['dev_invr'],
        'e'   : data['dev_energy'],
        'g'   : data['dev_gradient'],
        'n'   : data['dev_nac'],
        }
        self.miu_list={
        'invr': data['mid_invr'],
        'e'   : data['mid_energy'],
        'g'   : data['mid_gradient'],
        'n'   : data['mid_nac'],
        }

    def _heading(self):

        headline="""
%s
 *---------------------------------------------------*
 |                                                   |
 |            Gaussian Process Regression            |
 |                                                   |
 *---------------------------------------------------*

""" % (self.version)

       	return headline

    def _whatistime(self):
        return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

    def _howlong(self,start,end):
        walltime=end-start
        walltime='%5d days %5d hours %5d minutes %5d seconds' % (int(walltime/86400),int((walltime%86400)/3600),int(((walltime%86400)%3600)/60),int(((walltime%86400)%3600)%60))
        return walltime

    def _getinvr(self,x):
        x=np.array(x)
        x=np.array(x[:,1:],dtype=np.float) 
        invr=[]
        q=x[1:]
        for atom1 in x:
            for atom2 in q:
                d=np.sum((atom1-atom2)**2)**0.5
                invr.append(1/d)
            q=q[1:]
        invr=np.array(invr)
        return invr

    def train(self):

        start=time.time()
        topline='Gaussian Process Start: %20s\n%s' % (self._whatistime(),self._heading())
        runinfo="""\n  &gp fitting with %d threads\n""" % (self.ncpu)

        if self.silent == 0:
            print(topline)
            print(runinfo)

        log=open('%s.log' % (self.name),'w')
        log.write(topline)
        log.write(runinfo)
        log.close()

        self.model.fit(self.x,self.y_dict,n_processes=self.ncpu)
        self.model.save(f"fitted-{self.name}")
        y_pred,y_std=self.model.predict(self.x_val,n_processes=self.ncpu)
        length=len(self.x_val)

        ## todo make a loop later
        e_pred=y_pred['e']          *self.sgm_list['e']+self.miu_list['e']
        g_pred=y_pred['g']          *self.sgm_list['g']+self.miu_list['g']
        n_pred=y_pred['n']          *self.sgm_list['n']+self.miu_list['n']
        e_val=self.y_val_dict['e']  *self.sgm_list['e']+self.miu_list['e']
        g_val=self.y_val_dict['g']  *self.sgm_list['g']+self.miu_list['g']
        n_val=self.y_val_dict['n']  *self.sgm_list['n']+self.miu_list['n']

        e_dev=e_pred-e_val
        g_dev=g_pred-g_val
        n_dev=n_pred-n_val
        e_dev_max=np.amax(np.abs(e_dev))
        e_dev_min=np.amin(np.abs(e_dev))
        e_dev_rmsd=np.mean(e_dev**2)**0.5
        g_dev_max=np.amax(np.abs(g_dev))
        g_dev_min=np.amin(np.abs(g_dev))
        g_dev_rmsd=np.mean(g_dev**2)**0.5
        n_dev_max=np.amax(np.abs(n_dev))
        n_dev_min=np.amin(np.abs(n_dev))
        n_dev_rmsd=np.mean(n_dev**2)**0.5

        e_std=y_std['e']   *self.sgm_list['e']
        g_std=y_std['g']   *self.sgm_list['g']
        n_std=y_std['n']   *self.sgm_list['n']
        e_std_max=np.amax(e_std)
        e_std_min=np.amin(e_std)
        e_std_mean=np.mean(e_std)
        g_std_max=np.amax(g_std)
        g_std_min=np.amin(g_std)
        g_std_mean=np.mean(g_std)
        n_std_max=np.amax(n_std)
        n_std_min=np.amin(n_std)
        n_std_mean=np.mean(n_std)

        ## Here I will need some function to print/save output

        o=open('%s-e-errors.txt' % (self.name),'w')
        p=open('%s-g-errors.txt' % (self.name),'w')
        q=open('%s-nac-errors.txt' % (self.name),'w')

        e_vs=np.concatenate((e_pred.reshape([length,-1]),e_val.reshape([length,-1])),axis=1)
        g_vs=np.concatenate((g_pred.reshape([length,-1]),g_val.reshape([length,-1])),axis=1)
        n_vs=np.concatenate((n_pred.reshape([length,-1]),n_val.reshape([length,-1])),axis=1)

        np.savetxt(o,np.concatenate((e_vs,e_std.reshape([length,-1])),axis=1))
        np.savetxt(p,np.concatenate((g_vs,g_std.reshape([length,-1])),axis=1))
        np.savetxt(q,np.concatenate((n_vs,n_std.reshape([length,-1])),axis=1))
        o.close()
        p.close()
        q.close()

        train_info="""
  &gp results
-------------------------------------------------------
  energy     std max/std min/std mean: %16.8f %16.8f %16.8f
    Eh       dev max/dev min/dev rmsd: %16.8f %16.8f %16.8f

  gradient   std max/std min/std mean: %16.8f %16.8f %16.8f
  Eh/Bohr    dev max/dev min/dev rmsd: %16.8f %16.8f %16.8f

  nac        std max/std min/std mean: %16.8f %16.8f %16.8f
  1/Bohr     dev max/dev min/dev rmsd: %16.8f %16.8f %16.8f

""" % (e_std_max, e_std_min, e_std_mean, e_dev_max, e_dev_min, e_dev_rmsd,\
       g_std_max, g_std_min, g_std_mean, g_dev_max, g_dev_min, g_dev_rmsd,\
       n_std_max, n_std_min, n_std_mean, n_dev_max, n_dev_min, n_dev_rmsd)

        end=time.time()
        walltime=self._howlong(start,end)
        endline='Gaussian Process End: %20s Total: %20s\n' % (self._whatistime(),walltime)

        if self.silent == 0:
            print(train_info)
       	    print(endline)

        log=open('%s.log' % (self.name),'a')
        log.write(train_info)
        log.write(endline)
        log.close()

        return self

    def load(self):
        self.model.load(self.modelfile)

        return self

    def appendix(self,addons):
        ## fake function does nothing

        return self

    def evaluate(self,x):
        if x == None:
            with open('%s' % self.pred_data,'r') as preddata:
                pred=json.load(preddata)
            pred_natom,pred_nstate,pred_xyz,pred_invr,pred_energy,pred_gradient,pred_nac,pred_ci,pred_mo=pred
            x=pred_invr
        else:
            x=np.array([self._getinvr(x)])
        size=len(x)
        x=(x-self.miu_list['invr'])/self.sgm_list['invr']
        y_pred,y_std=self.model.predict(x)
       # print(x[0])
       # print(y_pred['g'][0])
        e_pred=y_pred['e'] *self.sgm_list['e']+self.miu_list['e']
        g_pred=y_pred['g'] *self.sgm_list['g']+self.miu_list['g']
        n_pred=y_pred['n'] *self.sgm_list['n']+self.miu_list['n']
        e_std=y_std['e']   *self.sgm_list['e']
        g_std=y_std['g']   *self.sgm_list['g']
        n_std=y_std['n']   *self.sgm_list['n']


        ## Here I will need some function to print/save output
        if self.silent == 0:
            o=open('%s-e.pred.txt' % (self.name),'w')
            p=open('%s-g.pred.txt' % (self.name),'w')
            q=open('%s-n.pred.txt' % (self.name),'w')
            length=len(x)
            np.savetxt(o,np.concatenate((e_pred.reshape([length,-1]),e_std.reshape([length,-1])),axis=1))
            np.savetxt(p,np.concatenate((g_pred.reshape([length,-1]),g_std.reshape([length,-1])),axis=1))
            np.savetxt(q,np.concatenate((n_pred.reshape([length,-1]),n_std.reshape([length,-1])),axis=1))
            o.close()
            p.close()
            q.close()

        ## fix return later
        return {
                'energy'   : e_pred.reshape([size,self.nstate])[0],
                'gradient' : g_pred.reshape([size,self.nstate,self.natom,3])[0],
                'nac'      : n_pred.reshape([size,self.npair,self.natom,3])[0],
                'civec'    : None,
                'movec'    : None,
                'err_e'    : e_std,
       	       	'err_g'	   : g_std,
       	       	'err_n'	   : n_std,
                }
