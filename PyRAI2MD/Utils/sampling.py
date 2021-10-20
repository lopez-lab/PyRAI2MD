######################################################
#
# PyRAI2MD 2 module for sampling initial condition from frequencies
#  (Molcas, G16, BAGEL, ORCA output)
# Author Jingbai Li
# May 21 2021
#
######################################################

import os, sys, random, tarfile
import numpy as np
from optparse import OptionParser
from numpy import linalg as la

class Element:
    ## This class is periodic table
    ## This class read atom name in various format
    ## This class return atomic properties

    def __init__(self,name):

        Periodic_Table = {
             "HYDROGEN":"1","H":"1","1":"1",
             "HELIUM":"2","He":"2","2":"2","HE":"2",
             "LITHIUM":"3","Li":"3","3":"3","LI":"3",
             "BERYLLIUM":"4","Be":"4","4":"4","BE":"4",
             "BORON":"5","B":"5","5":"5",
             "CARBON":"6","C":"6","6":"6",
             "NITROGEN":"7","N":"7","7":"7",
             "OXYGEN":"8","O":"8","8":"8",
             "FLUORINE":"9","F":"9","9":"9",
             "NEON":"10","Ne":"10","10":"10","NE":"10",
             "SODIUM":"11","Na":"11","11":"11","NA":"11",
             "MAGNESIUM":"12","Mg":"12","12":"12","MG":"12",
             "ALUMINUM":"13","Al":"13","13":"13","AL":"12",
             "SILICON":"14","Si":"14","14":"14","SI":"14",
             "PHOSPHORUS":"15","P":"15","15":"15",
             "SULFUR":"16","S":"16","16":"16",
             "CHLORINE":"17","Cl":"17","17":"17","CL":"17",
             "ARGON":"18","Ar":"18","18":"18","AG":"18",
             "POTASSIUM":"19","K":"19","19":"19",
             "CALCIUM":"20","Ca":"20","20":"20","CA":"20",
             "SCANDIUM":"21","Sc":"21","21":"21","SC":"21",
             "TITANIUM":"22","Ti":"22","22":"22","TI":"22",
             "VANADIUM":"23","V":"23","23":"23",
             "CHROMIUM":"24","Cr":"24","24":"24","CR":"24",
             "MANGANESE":"25","Mn":"25","25":"25","MN":"25",
             "IRON":"26","Fe":"26","26":"26","FE":"26",
             "COBALT":"27","Co":"27","27":"27","CO":"27",
             "NICKEL":"28","Ni":"28","28":"28","NI":"28",
             "COPPER":"29","Cu":"29","29":"29","CU":"29",
             "ZINC":"30","Zn":"30","30":"30","ZN":"30",
             "GALLIUM":"31","Ga":"31","31":"31","GA":"31",
             "GERMANIUM":"32","Ge":"32","32":"32","GE":"32",
             "ARSENIC":"33","As":"33","33":"33","AS":"33",
             "SELENIUM":"34","Se":"34","34":"34","SE":"34",
             "BROMINE":"35","Br":"35","35":"35","BR":"35",
             "KRYPTON":"36","Kr":"36","36":"36","KR":"36",
             "RUBIDIUM":"37","Rb":"37","37":"37","RB":"37",
             "STRONTIUM":"38","Sr":"38","38":"38","SR":"38",
             "YTTRIUM":"39","Y":"39","39":"39",
             "ZIRCONIUM":"40","Zr":"40","40":"40","ZR":"40",
             "NIOBIUM":"41","Nb":"41","41":"41","NB":"41",
             "MOLYBDENUM":"42","Mo":"42","42":"42","MO":"42",
             "TECHNETIUM":"43","Tc":"43","43":"43","TC":"43",
             "RUTHENIUM":"44","Ru":"44","44":"44","RU":"44",
             "RHODIUM":"45","Rh":"45","45":"45","RH":"45",
             "PALLADIUM":"46","Pd":"46","46":"46","PD":"46",
             "SILVER":"47","Ag":"47","47":"47","AG":"47",
             "CADMIUM":"48","Cd":"48","48":"48","CD":"48",
             "INDIUM":"49","In":"49","49":"49","IN":"49",
             "TIN":"50","Sn":"50","50":"50","SN":"50",
             "ANTIMONY":"51","Sb":"51","51":"51","SB":"51",
             "TELLURIUM":"52","Te":"52","52":"52","TE":"52",
             "IODINE":"53","I":"53","53":"53",
             "XENON":"54","Xe":"54","54":"54","XE":"54",
             "CESIUM":"55","Cs":"55","55":"55","CS":"55",
             "BARIUM":"56","Ba":"56","56":"56","BA":"56",
             "LANTHANUM":"57","La":"57","57":"57","LA":"57",
             "CERIUM":"58","Ce":"58","58":"58","CE":"58", 
             "PRASEODYMIUM":"59","Pr":"59","59":"59","PR":"59",
             "NEODYMIUM":"60","Nd":"60","60":"60","ND":"60", 
             "PROMETHIUM":"61","Pm":"61","61":"61","PM":"61", 
             "SAMARIUM":"62","Sm":"62","62":"62","SM":"62",
             "EUROPIUM":"63","Eu":"63","63":"63","EU":"63", 
             "GADOLINIUM":"64","Gd":"64","64":"64","GD":"64", 
             "TERBIUM":"65","Tb":"65","65":"65","TB":"65",
             "DYSPROSIUM":"66","Dy":"66","66":"66","DY":"66", 
             "HOLMIUM":"67","Ho":"67","67":"67","HO":"67", 
             "ERBIUM":"68","Er":"68","68":"68","ER":"68", 
             "THULIUM":"69","TM":"69","69":"69","TM":"69", 
             "YTTERBIUM":"70","Yb":"70","70":"70","YB":"70", 
             "LUTETIUM":"71","Lu":"71","71":"71","LU":"71",
             "HAFNIUM":"72","Hf":"72","72":"72","HF":"72",
             "TANTALUM":"73","Ta":"73","73":"73","TA":"73",
             "TUNGSTEN":"74","W":"74","74":"74",
             "RHENIUM":"75","Re":"75","75":"75","RE":"75",
             "OSMIUM":"76","Os":"76","76":"76","OS":"76",
             "IRIDIUM":"77","Ir":"77","77":"77","IR":"77",
             "PLATINUM":"78","Pt":"78","78":"78","PT":"78",
             "GOLD":"79","Au":"79","79":"79","AU":"79",
             "MERCURY":"80","Hg":"80","80":"80","HG":"80",
             "THALLIUM":"81","Tl":"81","81":"81","TL":"81",
             "LEAD":"82","Pb":"82","82":"82","PB":"82",
             "BISMUTH":"83","Bi":"83","83":"83","BI":"83",
             "POLONIUM":"84","Po":"84","84":"84","PO":"84",
             "ASTATINE":"85","At":"85","85":"85","AT":"85",
             "RADON":"86","Rn":"86","86":"86","RN":"86"}

        FullName=["HYDROGEN", "HELIUM", "LITHIUM", "BERYLLIUM", "BORON", "CARBON", "NITROGEN", "OXYGEN", "FLUORINE", "NEON", 
              "SODIUM", "MAGNESIUM", "ALUMINUM", "SILICON", "PHOSPHORUS", "SULFUR", "CHLORINE", "ARGON", "POTASSIUM", "CALCIUM", 
              "SCANDIUM", "TITANIUM", "VANADIUM", "CHROMIUM", "MANGANESE", "IRON", "COBALT", "NICKEL", "COPPER", "ZINC", 
              "GALLIUM", "GERMANIUM", "ARSENIC", "SELENIUM", "BROMINE", "KRYPTON", "RUBIDIUM", "STRONTIUM", "YTTRIUM", "ZIRCONIUM", 
              "NIOBIUM", "MOLYBDENUM", "TECHNETIUM", "RUTHENIUM", "RHODIUM", "PALLADIUM", "SILVER", "CADMIUM", "INDIUM", "TIN", 
              "ANTIMONY", "TELLURIUM", "IODINE", "XENON", "CESIUM", "BARIUM", "LANTHANUM", "CERIUM", "PRASEODYMIUM", "NEODYMIUM", 
              "PROMETHIUM", "SAMARIUM", "EUROPIUM", "GADOLINIUM", "TERBIUM", "DYSPROSIUM", "HOLMIUM", "ERBIUM", "THULIUM", "YTTERBIUM", 
              "LUTETIUM", "HAFNIUM", "TANTALUM", "TUNGSTEN", "RHENIUM", "OSMIUM", "IRIDIUM", "PLATINUM", "GOLD", "MERCURY", 
              "THALLIUM", "LEAD", "BISMUTH", "POLONIUM", "ASTATINE", "RADON"]

        Symbol=[ "H","He","Li","Be","B","C","N","O","F","Ne",
                "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
                "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
                "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
                "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
                "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
                "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","TM","Yb",
                "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
                "Tl","Pb","Bi","Po","At","Rn"]

        Mass=[1.008,4.003,6.941,9.012,10.811,12.011,14.007,15.999,18.998,20.180,
              22.990,24.305,26.982,28.086,30.974,32.065,35.453,39.948,39.098,40.078,
              44.956,47.867,50.942,51.996,54.938,55.845,58.933,58.693,63.546,65.390,
              69.723,72.640,74.922,78.960,79.904,83.800,85.468,87.620,88.906,91.224,
              92.906,95.940,98.000,101.070,102.906,106.420,107.868,112.411,114.818,118.710,
              121.760,127.600,126.905,131.293,132.906,137.327,138.906,140.116,140.908,144.240,
              145.000,150.360,151.964,157.250,158.925,162.500,164.930,167.259,168.934,173.040,
              174.967,178.490,180.948,183.840,186.207,190.230,192.217,195.078,196.967,200.590,
              204.383,207.200,208.980,209.000,210.000,222.000]

        # Van der Waals Radius, missing data replaced by 2.00
        Radii=[1.20,1.40,1.82,1.53,1.92,1.70,1.55,1.52,1.47,1.54,
               2.27,1.73,1.84,2.10,1.80,1.80,1.75,1.88,2.75,2.31,
               2.11,2.00,2.00,2.00,2.00,2.00,2.00,1.63,1.40,1.39,
               1.87,2.11,1.85,1.90,1.85,2.02,3.03,2.49,2.00,2.00,
               2.00,2.00,2.00,2.00,2.00,1.63,1.72,1.58,1.93,2.17,
               2.00,2.06,1.98,2.16,3.43,2.68,2.00,2.00,2.00,2.00,
               2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,
               2.00,2.00,2.00,2.00,2.00,2.00,2.00,1.75,1.66,1.55,
               1.96,2.02,2.07,1.97,2.02,2.20]

        self.__name = int(Periodic_Table[name])
        self.__FullName = FullName[self.__name-1]
        self.__Symbol = Symbol[self.__name-1]
        self.__Mass = Mass[self.__name-1]
        self.__Radii = Radii[self.__name-1]

    def getFullName(self):
        return self.__FullName
    def getSymbol(self):
        return self.__Symbol
    def getUpperSymbol(self):
        return self.__Symbol.upper()
    def getMass(self):
        return self.__Mass
    def getNuc(self):
        return self.__name
    def getNelectron(self):
        return self.__name
    def getRadii(self):
        return self.__Radii

def ReadMolden(input):
    ## This function read Molcas .freq.molden file and return all data as a dict
    ## The freqdata has clear descriptions for all used variables
    ## The Molcas saves unnormalized and unmass-weighted normal modes

    with open('%s.freq.molden' % (input),'r') as molden:
        file = molden.read().splitlines()
        n = 0
        modes=[]
        for line in file:
            n += 1
            if """ [N_FREQ]""" in line:
                nmode = int(file[n])
            if """ [FREQ]""" in line:
       	       	freqs = [i.split() for i in file[n: n + nmode]]
       	       	freqs = np.array(freqs).astype(float)
                for i in freqs:
                    if i < 0:
                        print('imaginary frequency: %f' % i)
                freqs = np.absolute(freqs)
            if """ [INT]""" in line:
                inten = [i.split() for i in file[n: n + nmode]]
                inten = np.array(inten).astype(float)
            if """ [NATOM]""" in line:
                natom = int(file[n])
            if """ [FR-COORD]""" in line:
                coord = [i.split() for i in file[n: n + natom]]
                coord = np.array(coord)
                atoms, xyz = coord[:, 0], coord[:, 1:].astype(float)
                amass = [Element(atm).getMass() for atm in atoms]
                amass = np.array(amass)
                achrg = [Element(atm).getNuc() for atm in atoms]
                achrg = np.array(achrg)
            if """ vibration""" in line:
                vib = [i.split() for i in file[n: n + natom]]
                vib = np.array(vib).astype(float)
                modes.append(vib)
            if """ [RMASS]""" in line:
                rmass = [float(i) for i in file[n: n + nmode]]
                rmass = np.array(rmass)

    freqs = freqs.reshape((nmode, 1))
    amass = amass.reshape((natom, 1))
    achrg = achrg.reshape((natom, 1))
    modes = np.array(modes).reshape((nmode, natom, 3))
    rmass = rmass.reshape((nmode, 1))

    freqdata = {
    'nfreq':nmode, # number of degrees
    'freqs':freqs, # frequencies in cm-1, [N of dgree[x]]
    'inten':inten, # ir intensity
    'natom':natom, # number of atoms
    'atoms':atoms, # atom list
    'xyz':xyz,     # cartesian coordinates, [N of atoms [x,y,z]]
    'vib':modes,   # vibrations	in cartesian, [N of degrees [N of atoms	[x,y,z]]]
    'rmass':rmass, # reduced masses, [N of degree [x]]
    'amass':amass, # atomic masses, [N of atoms [x]]
    'achrg':achrg, # atomic charges, [N of atoms [x]]
    }

    return freqdata

def ReadG16(input):
    ## This function read .freq.g16 file (Gaussian .log file) and .freq.fchk and return all data as a dict
    ## The freqdata has clear descriptions for all used variables
    ## The G16 saves normalized unmass-weighted normal modes


    with open('%s.freq.fchk' % (input),'r') as raw:
        fchk = raw.read().splitlines()
    with open('%s.freq.g16' % (input),'r') as raw:
        log = raw.read().splitlines()

    ## extracting data from log
    freqs_list = []
    rmass_list = []
    inten_list = []

    for line in log:
        if 'Frequencies' in line:
            f = line.split(' -- ')[-1]
            freqs_list.append(f)

        if 'Red. masses' in line:
            r = line.split(' -- ')[-1]
            rmass_list.append(r)

        if 'IR Inten' in line:
            i = line.split(' -- ')[-1]
            inten_list.append(i)

    freqs = G16format(freqs_list,[-1, 1])
    rmass = G16format(rmass_list,[-1, 1])
    inten = G16format(inten_list,[-1])

    ## extracting data from fchk

    natom = 0
    atom_line = 0
    atom_list = []
    ncart = 0
    ncart_line = 0
    cart_list = []
    nmode = 0
    nvect = 0
    vect_line = 0
    vect_list = []

    for n, line in enumerate(fchk):
        if 'Atomic numbers' in line:
            natom = int(line.split()[-1])
            atom_line = int(natom / 6) + (natom % 6 > 0)
            atom_list = fchk[n + 1: n + 1 + atom_line]

        if 'Current cartesian coordinates' in line:
            ncart = int(line.split()[-1])
            cart_line = int(ncart / 5) + (ncart % 5 > 0)
            cart_list = fchk[n + 1: n + 1 + cart_line]

        if 'Number of Normal Modes' in line:
            nmode = int(line.split()[-1])

        if 'Vib-Modes' in line:
            nvect = int(line.split()[-1])
            vect_line = int(nvect / 5) + (nvect % 5 > 0)
            vect_list = fchk[n + 1: n + 1 + vect_line]

    atoms = G16format(atom_list, [-1])
    atoms = np.array([Element(str(int(i))).getSymbol() for i in atoms])
    xyz = G16format(cart_list, [natom, 3])
    modes = G16format(vect_list, [nmode,natom, 3])

    amass = [Element(i).getMass() for i in atoms]
    amass = np.array(amass)
    amass = amass.reshape((natom, 1))

    achrg = [Element(i).getNuc()  for i in atoms]
    achrg = np.array(achrg)
    achrg = achrg.reshape((natom, 1))

    modes = np.array([i / la.norm(i * amass**0.5) for i in modes]) # convert to unnormalized unmass-weighted

    freqdata={
    'nfreq':nmode, # number of degrees
    'freqs':freqs, # frequencies in cm-1, [N of dgree[x]]
    'inten':inten, # ir intensity
    'natom':natom, # number of atoms
    'atoms':atoms, # atom list
    'xyz':xyz,     # cartesian coordinates, [N of atoms [x,y,z]]
    'vib':modes,   # vibrations in cartesian, [N of degrees [N of atoms [x,y,z]]]
    'rmass':rmass, # reduced masses, [N of degree [x]]
    'amass':amass, # atomic masses, [N of atoms [x]]
    'achrg':achrg, # atomic charges, [N of atoms [x]]
    }

    return freqdata

def G16format(data,dshape):
    ## formatting data
    dlist = []
    for i in data:
        dlist += [float(x) for x in i.split()]
    dlist = np.array(dlist)
    dlist = dlist.reshape(dshape)

    return dlist

def ReadOrca(input):
    ## This function read ORCA .hess file and return all data as a dict
    ## The freqdata has clear descriptions for all used variables
    ## The ORCA saves normalized unmass-weighted normal modes

    with open('%s.freq.orca' % (input),'r') as molden:
        hess = molden.read().splitlines()

    ## extracting data from hess
    natom = 0
    for n, line in enumerate(hess):
        if '$vibrational_frequencies' in line:
            natom = int(int(hess[n + 1]) / 3)
            f = hess[n + 2: n + 2 + natom]
            freqs = np.array([x.split() for x in f]).astype(float)[:, 1].reshape((-1, 1))
        if '$ir_spectrum' in line:
       	    natom = int(int(hess[n + 1]) / 3)
       	    i = hess[n + 2 :n + 2 + natom]
       	    inten = np.array([x.split() for x in i]).astype(float)[:, 1].reshape((-1, 1))
        if '$atoms' in line:
            natom = int(hess[n + 1])
            coord = hess[n + 2: n + 2 + natom]
            coord = np.array([x.split() for x in coord])
            atoms = coord[:, 0]
            xyz = coord[:, 2: 5].astype(float)
        if '$normal_modes' in line:
            nmode = int(hess[n + 1].split()[0])  
            nline = (nmode + 1) * (int(nmode / 5) + (nmode % 5 > 0))
            vects = hess[n + 2: n + 2 + nline]
            modes = [[] for i in range(nmode)]
            for n,i in enumerate(vects):
                row=(n)%(nmode+1)-1 
                if row >= 0:
                    modes[row] += [float(j) for j in i.split()[1:]]
            modes=np.array(modes).T.reshape((nmode,int(nmode / 3),3))  # Transpose array !!!

    natom = len(atoms)
    # filter out imaginary and trans-rot freqs and modes
    realfreq = []
    for n,i in enumerate(freqs):
        if   i > 0:
            realfreq.append(n)
        elif i < 0:
            print('imaginary frequency: %10f mode: %6d (Ingored)' % (i,n+1))

    nmode = len(realfreq)
    freqs = freqs[realfreq].reshape((-1, 1))
    inten = inten[realfreq]
    modes = modes[realfreq]

    rmass = np.array([0 for i in range(nmode)]).reshape((-1, 1))
    amass = [Element(atm).getMass() for atm in atoms]
    amass = np.array(amass)
    amass = amass.reshape((natom,1))

    achrg = [Element(atm).getNuc() for atm in atoms]
    achrg = np.array(achrg)
    achrg = achrg.reshape((natom, 1))

    modes = np.array([i / la.norm(i * amass**0.5) for i in modes]) # convert to unnormalized unmass-weighted

    freqdata={
    'nfreq':nmode, # number of degrees
    'freqs':freqs, # frequencies in cm-1, [N of dgree[x]]
    'inten':inten, # ir intensity
    'natom':natom, # number of atoms
    'atoms':atoms, # atom list
    'xyz':xyz,     # cartesian coordinates, [N of atoms [x,y,z]]
    'vib':modes,   # vibrations	in cartesian, [N of degrees [N of atoms	[x,y,z]]]
    'rmass':rmass, # reduced masses, [N of degree [x]]
    'amass':amass, # atomic masses, [N of atoms [x]]
    'achrg':achrg, # atomic charges, [N of atoms [x]]
    }

    return freqdata

def ReadBagel(input):
    ## This function read .freq.bagel file (BAGEL .log file) and return all data as a dict
    ## The freqdata has clear descriptions for all used variables
    ## The BAGEL saves normalized mass-weighted normal modes

    with open('%s.freq.bagel' % (input),'r') as raw:
        log = raw.read().splitlines()

    natom = 0
    atoms = []
    xyz = []
    cart_list = []
    nmode = 0
    nvect = 0
    vect_line = 0
    vect_list = []
    freqs = []
    inten = []

    for n, line in enumerate(log):
        if 'Freq (cm-1)' in line:
            freqs += [float(i) for i in line.split()[2:]]
        if 'IR Int. (km/mol)' in line:
            inten += [float(i) for i in line.split()[3:]]
        if '"atom"' in line:
            natom += 1
            nmode = 3 * natom
            line = line.replace(',',' ').replace('"',' ').split()
            atoms.append(line[3])
            xyz.append([float(i) for i in line[7: 10]])
        if '++ Mass Weighted Hessian Eigenvectors ++' in line:
            vect_line = (nmode + 2) * (int(nmode / 6) + (nmode % 6 > 0))
            vect_list = log[n + 1: n + 1 + vect_line]


    modes = [[] for i in range(nmode)]
    for n, i in enumerate(vect_list):
        row = (n) % (nmode + 2) -2 
        if (n) % (nmode + 2) > 1:
            modes[row] += [float(j) for j in i.split()[1:]]

    atoms = np.array(atoms)
    freqs = np.array(freqs)
    inten = np.array(inten)
    xyz = np.array(xyz)
    modes = np.array(modes).T.reshape((nmode, natom, 3))  # Transpose array !!!

    # filter out imaginary and trans-rot freqs and modes
    realfreq=[]
    for n, i in enumerate(freqs):
        if   i > 0:
            realfreq.append(n)
        elif i < 0:
            print('imaginary frequency: %10f mode: %6d (Ingored)' % (i, n + 1))

    nmode = len(realfreq)
    freqs = freqs[realfreq].reshape((-1, 1))
    inten = inten[realfreq]
    modes = modes[realfreq]

    rmass = np.array([0 for i in range(nmode)]).reshape((-1, 1))
    amass = [Element(atm).getMass() for atm in atoms]
    amass = np.array(amass)
    amass = amass.reshape((natom, 1))

    achrg = [Element(atm).getNuc() for atm in atoms]
    achrg = np.array(achrg)
    achrg = achrg.reshape((natom, 1))

    modes = np.array([i /amass**0.5 for i in modes]) # convert to unnormalized unmass-weighted

    freqdata={
    'nfreq':nmode, # number of degrees
    'freqs':freqs, # frequencies in cm-1, [N of dgree[x]]
    'inten':inten, # ir intensity
    'natom':natom, # number of atoms
    'atoms':atoms, # atom list
    'xyz':xyz,     # cartesian coordinates, [N of atoms [x,y,z]]
    'vib':modes,   # vibrations in cartesian, [N of degrees [N of atoms [x,y,z]]]
    'rmass':rmass, # reduced masses, [N of degree [x]]
    'amass':amass, # atomic masses, [N of atoms [x]]
    'achrg':achrg, # atomic charges, [N of atoms [x]]
    }


    return freqdata

def LoadNewtonX(input):
    ## This function read .init.newtonx file (NewtonX final_output file) and return all data as a dict
    ## This function doesn't do initial condition sampling

    with open('%s.init.newtonx' % (input),'r') as initcond:
        nx = initcond.read().splitlines()

    start = 0
    end = 0
    ensemble = []
    for n,i in enumerate(nx):
        if   'Geometry' in i:
            start = n
        elif 'Velocity' in i:
            end = n
            break

    natom = end - start - 1

    for n, i in enumerate(nx):
        if 'Initial condition' in i:
            coord = [i.split() for i in nx[n + 2: n + 2 + natom]]
            coord = np.array(coord)
            xyz = coord[:, [0, 2, 3, 4]]
            mass = coord[:, [5, 1]]
            veloc = [i.split() for i in nx[n + 3 + natom: n + 3 + natom * 2]]
            veloc = np.array(veloc)
            initcond = np.concatenate((xyz, veloc), axis = 1)
            initcond = np.concatenate((initcond, mass), axis = 1)
            ensemble.append(initcond)

    return ensemble

def LoadXYZ(input):
    ## This function read .init.xyz file (Gen-FSSH.py .init file) and return all data as a list
    ## This function doesn't do initial condition sampling

    with open('%s.init.xyz' % (input),'r') as initcond:
        gen=initcond.read().splitlines()

    ensemble = []
    for n,i in enumerate(gen):
        if 'Init' in i:
            natom = int(i.split()[2])
            initcond = [i.split() for i in gen[n + 1: n + 1 + natom]]
            initcond = np.array(initcond)
            ensemble.append(initcond)

    return ensemble

def LoadXZ(input):
    ## This function read .init.tar.xz file and return all as a list of dict

    with tarfile.open('%s.init.tar.xz' % (input), 'r:xz') as initcond:
        name = initcond.getnames()[0]
        file = initcond.extractfile(name)
        data = file.read().decode().splitlines()
    ensemble=[]
    for n, i in enumerate(data):
        if 'Init' in i:
            nxyz = int(i.split()[2])
            nvelo = int(i.split()[3])
            xyz = data[n + 1: n + 1 + nxyz]
            velo = data[n + 1 + nxyz: n + 1 + nxyz + nvelo]

            ensemble.append({'txyz': xyz, 'velo': velo})

    return ensemble

def Gaussian():
    ## This function generates standard normal distribution variates from a random number in uniform distribution
    ## This function is used for Boltzmann sampling
    ## This function returns a coefficient to update structure or velocity

    u1 = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
    z = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2) # Box-Muller transformation
    return float(z)

def Boltzmann(sample):
    ## This function is based on Molcas dynamixtool.py
    ## This function does Boltzmann sampling for structure and velocity
    ## This function calls Gaussian to find update coefficient
    ## This function returns initial condition as [[atom x y z v(x) v(y) v(z)],...]

    temp = sample['temp']
    nfreq = sample['nfreq']
    freqs = sample['freqs']
    natom = sample['natom']
    atoms = sample['atoms']
    xyz = sample['xyz']
    vib = sample['vib']
    rmass = sample['rmass']
    amass = sample['amass']
    achrg = sample['achrg']

    mu_to_hartree    = 4.55633518e-6   # 1 cm-1  = h*c/Eh       = 4.55633518e-6
    ma_to_amu        = 1822.88852      # 1 g/mol = 1/Na*me*1000 = 1822.88852 amu
    bohr_to_angstrom = 0.529177249     # 1 Bohr  = 0.529177249 Angstrom
    k_to_au          = 3.16681156e-6   # 1 K     = 3.16681156e-6 au of temperature

    ## Constants:
    ##            meaning                     unit             conversion
    ## kb         Boltzmann constant
    ## h          Planck constant             [kg*m^2/s]
    ## h_bar	  reduced Planck constant     [kg*m^2/s]       h_bar = h / (2*pi)
    ## a0         Bohr radii                  [m]              a0=sqrt(Eh/h_bar^2)
    ## Eh         Hartree energy              [kg*m^2/s^2]     Eh=h_bar^2/(me*a0^2)
    ## Na         Avogadro's number
    ## ma         molar mass                  [g/mol]
    ## m          atomic mass                 [kg]
    ## me         electron mass               [kg]
    ## c          speed of light              [m/s]
    ## v          velocity                    [m/s]            v=Eh/me*vau
    ## vau        velocity in atomic unit     [Bohr/au]
    ## w          angluar frequency           [s-1]            w=2*pi*nu
    ## nu         frequency                   [s-1]            nu = mu * c
    ## mu         waveunmber                  [cm-1]           mu = nu / c
    ## T          temperature                 [K]              T = Eh/kb
    ## vib        normal modes                [Bohr]           Molcas prints out non-weighted coordinates in Bohr
    ## n          mass-weighted normal modes  [Bohr*amu^0.5]   n=vib*sqrt(m)
    ## R/R0	  cartesian coordinate        [Bohr]
    ## Q/P        dimensionless coordinates and velocity
    ##
    ## Math note:
    ##
    ## Convert wavenumber[cm-1]to energy[au] E=h*nu=h*c*mu=mu*(h*c/Eh)*Eh
    ## Convert molar mass[g/mol] to atomic mass unit m=ma/Na*me*1000
    ##
    ## sigma_Q = sqrt(kb*T/m)/w    standard deviation of postion in Boltzmann distribution
    ## sigma_P = sqrt(kb*T/m)      standard deviation of velocity in Boltzmann distribution
    ## 
    ## position q = sigma_Q*Q, velocity v = sigma_P*Q
    ## 
    ## Update position R from R0 and velocity v from 0
    ## R = R0+sum(sum(q*n)*mode), v = sum(sum(v*n)*mode)
    ##
    ## Convert position [m] to atomic unit [Bohr]
    ## q = sqrt(kb*T/m)/w
    ##   = sqrt(T/(Eh/kb))/[ sqrt(m/me) * w*h_bar]*sqrt(Eh*h_bar^2/me) 
    ##   = sqrt(T/(Eh/kb))/[ sqrt(m/me)*mu*(h*c/Eh) ]*sqrt(h_bar^2/me*Eh)
    ##   = sqrt(T/(Eh/kb))/[ sqrt(m/me)*mu*(h*c/Eh) ]*a0 
    ##   = sqrt(T/(Eh/kb))/[ sqrt(m/me)*mu*(h*c/Eh) ]in [Bohr]
    ##
    ## Convert velocity[m/s] to atomic unit[Bohr/au]
    ## v = sqrt(kb*T/m) = sqrt(T/(Eh/kb)/sqrt(m/me))*sqrt(Eh/me) 
    ##   = sqrt(T/(Eh/kb)/sqrt(m/me))*vau
    ##   = sqrt(T/(Eh/kb)/sqrt(m/me)) in [Bohr/au]

    amass = np.array([np.ones(3) * i for i in amass]) # expand atoic mass over x y z coordinates

    sigma_Q = np.sqrt(temp * k_to_au) / np.sqrt(freqs * mu_to_hartree) #standard deviation of mass-weighted position
    sigma_P = np.sqrt(temp * k_to_au)                                  #standard deviation of mass-weighted velocity
    Q_P = np.array([[Gaussian(),Gaussian()] for i in freqs])           #generates update coordinates and momenta pairs Q and P

    Q = Q_P[:, 0].reshape((nfreq,1))                       # first column is Q
    Q *= sigma_Q                                           # project standard normal distribution back to position space
    Qvib = np.array([np.ones((natom, 3))*i for i in Q])    # generate identity array to expand Q
    Qvib = np.sum(vib * Qvib, axis = 0)                    # sum mass-weighted position over all modes
    Qvib /= np.sqrt(amass * ma_to_amu)                     # un-weight position in Bohr
    newc = (xyz + Qvib) * bohr_to_angstrom                 # cartesian coordinates in Angstrom

    P = Q_P[:, 1].reshape((nfreq, 1))                      # second column is P
    P *= sigma_P                                           # convert velocity from m/s to Bohr/au
    Pvib = np.array([np.ones((natom, 3))*i for i in P])    # generate identity array to expand P
    Pvib = np.sum(vib * Pvib, axis = 0)                    # sum mass-weighted velocity over all modes
    velo = Pvib/np.sqrt(amass*ma_to_amu)                   # un-weight velocity in Bohr/au
                                                           # use velo=1 in &DYNAMIX to read un-weighted velocity
    #These lines are for test
    #print(np.linalg.norm(vib[0]*amass**0.5))              # check if mode is mass-weighted (norm=1)  
    #w2=np.sum([(i*4.559489488e-06)**2 for i in freqs])    # 1 cm-1 = 2*pi*1*100*3*10^8*2.4188843e-17 au-1
    #Epot=0.5*np.sum(amass*ma_to_amu*w2*Qvib**2) 
    #Ekin=0.5*np.sum(amass*ma_to_amu*velo**2)         
    #print(Epot,Ekin,Epot+Ekin)

    #velo*=np.sqrt(amass*ma_to_amu)                        # only use for velo=2 in &DYNAMIX 
                                                           # convert velocity[Bohr/au] to mass-weighted [Bohr*amu^0.5/au] for Molcas

    inicond = np.concatenate((newc, velo), axis = 1)
    inicond = np.concatenate((atoms.reshape((-1, 1)), inicond), axis = 1)
    inicond = np.concatenate((inicond, amass[:, 0: 1]), axis = 1)
    inicond = np.concatenate((inicond, achrg), axis = 1)
    return inicond

def Laguerre(n, x):
    ## This function calculates laguerre polynomial
    ## L = n!/[(n-m)! * (m!)**2] = n*(n-1)*...*(n-m+1)/(m!)**2 = n/1**2 * (n-1)/2**2 *...*(n-m+1)/m**2, 0 <= m <= n

    L = 1 # L=1 when m=0   
    for m in range(1, n + 1):
        r = 1
        for mm in range(1, m + 1):
            r *= float(n - mm + 1) / mm**2
        L += (-1)**m * r * x**m

    return L


def Wignerfunc(mu, temp):
    ## This function generates random position Q and momenta P to find uptdate coifficents
    ## This function calls Laguerre to calculate the polynomial
    ## This function returns accepted Q and P

    #print('\nFreq: %s\n' % mu)
    max_pop = 0.9999
    ex = mu / (0.69503 * temp) #vibrational temperature: ex=h*c*mu/(kb*T), 0.69503 convert cm-1 to K
    pop= 0
    lvl_pop = []
    n = -1
    while True:
        n += 1
        pop += np.exp(-1 * ex * n) * (1 - np.exp(-1 * ex))
        lvl_pop.append(pop[0]) #Note here pop is a numpy array, thus pop[0] is the float number
        # Here is how I obtained this equation:
        # calculate partion function, fP=np.exp(ex*-0.5) /( 1 - np.exp(ex*-1) )
        # calculate population, pop=np.exp(-1*ex*(n+0.5))/fP
        #print('wignerfunction:%d %f %f %f %f'%(n,ex,np.exp(-1*ex*n)*(1-np.exp(-1*ex)),pop,max_pop))
        if pop >= max_pop:
            break
    while True:
        random_state=random.uniform(0,pop) # random generate a state
        n = -1
        for i in lvl_pop:                  # but population is not uniformly distributed over states
            n += 1
            if random_state <= i:          # find the lowest state that has more population than the random state
                break
        Q = random.uniform(0,1)*10.0-5.0
        P = random.uniform(0,1)*10.0-5.0
        rho2 = 2 * (Q**2 + P**2)
        W = (-1)**n * Laguerre(n, rho2) * np.exp(-0.5 * rho2)
        R=random.uniform(0, 1)
        #print('N: %d Q: %f P: %f W: %f R: %f' % (n,Q,P,W,R))
        if W > R and W < 1:
            #print('N: %d Q: %f P: %f Rho^2: %f W: %f R: %f' % (n,Q,P,rho2/2,W,R))

            break
    
    return float(Q), float(P)

def Wigner(sample):
    ## This function is based on SHARC wigner.py
    ## This function does Wigner sampling for structure and velocity
    ## This function calls Wiguerfunc to find update coefficient
    ## This function returns initial condition as [[atom x y z v(x) v(y) v(z)],...]

    temp = sample['temp']
    nfreq = sample['nfreq']
    freqs = sample['freqs']
    natom = sample['natom']
    atoms = sample['atoms']
    xyz = sample['xyz']
    vib = sample['vib']
    rmass = sample['rmass']
    amass = sample['amass']
    achrg = sample['achrg']

    mu_to_hartree    = 4.55633518e-6   # 1 cm-1  = h*c/Eh = 4.55633518e-6 au
    ma_to_amu        = 1822.88852      # 1 g/mol = 1/Na*me*1000 = 1822.88852 amu
    bohr_to_angstrom = 0.529177249     # 1 Bohr  = 0.529177249 Angstrom

    ## Constants:
    ##            meaning                     unit             conversion
    ## h          Planck constant             [kg*m^2/s]
    ## h_bar	  reduced Planck constant     [kg*m^2/s]       h_bar = h / (2*pi)
    ## a0         Bohr radii                  [m]              a0=sqrt(Eh/h_bar^2)
    ## Eh         Hartree energy              [kg*m^2/s^2]     Eh=h_bar^2/(me*a0^2)
    ## Na         Avogadro's number           
    ## ma         molar mass                  [g/mol]
    ## m          atomic mass                 [kg]
    ## me         electron mass               [kg]
    ## c          speed of light              [m/s]
    ## v          velocity                    [m/s]            v=Eh/me*vau
    ## vau        velocity in atomic unit     [Bohr/au]
    ## w          angluar frequency           [s-1]            w=2*pi*nu
    ## nu         frequency                   [s-1]            nu = mu * c
    ## mu         waveunmber                  [cm-1]           mu = nu / c
    ## vib        normal modes                [Bohr]           Molcas prints out non-weighted coordinates in Bohr
    ## n          mass-weighted normal modes  [Bohr*amu^0.5]   n=vib*sqrt(m)
    ## R/R0       cartesian coordinate        [Bohr]
    ## Q/P        dimensionless coordinates and momenta
    ##
    ## Math note:
    ##
    ## Convert wavenumber[cm-1]to energy[au] E=h*nu=h*c*mu=mu*(h*c/Eh)*Eh
    ## Convert molar mass[g/mol] to atomic mass unit m=ma/Na*me*1000
    ##
    ## Convert position q[m] to atomic distance[Bohr]
    ## Q = sqrt(w*m/h_bar)*q => q = Q*sqrt(h_bar/w*m) = Q*sqrt(h_bar^2/w*h_bar*m) = Q*sqrt(h_bar^2/Eh*me)/[ sqrt(E/Eh)*sqrt(m/me) ] = Q*a0/[ sqrt(mu*(h*c/Eh))*sqrt(ma/(Na*me*1000)) ]
    ## q = Q/[ sqrt(mu*(h*c/Eh))*sqrt(ma/(Na*me*1000)) ] in [Bohr]
    ## Update position R from R0
    ## R = R0+sum(sum(q*n)*mode)
    ## q*n = Q/[ sqrt(mu*(h*c/Eh))*sqrt(ma/(Na*me*1000)) ]*vib*sqrt(m) = Q/sqrt(mu*(h*c/Eh))*sqrt(Na*me*1000)*vib
    ## q*n = Q/sqrt(mu*mu_to_hartree)*sqrt(1/ma_to_amu)*vib
    ##
    ## Convert velocity[m/s] to atomic unit[Bohr/au]
    ## P = p/sqrt(w*m*h_bar) => p = m*v = P*sqrt(w*m*h_bar) => v = P*sqrt(w*h_bar/m) = P*sqrt(Eh/me)*sqrt(E/Eh)/sqrt(m/me) = P*vau*sqrt(mu*(h*c/Eh))/sqrt(ma/Na*me*1000))
    ## v = P*sqrt(mu*(h*c/Eh))/sqrt(ma/Na*me*1000)) in [Bohr/au]
    ## Update velcocity from 0 to v 
    ## v = 0+sum(sum(p*n)*mode)
    ## v*n = P*sqrt*(mu*(h*c/Eh))/sqrt(ma/(Na*me*1000))*vib*sqrt(m) = P*sqrt(mu*(h*c/Eh))*sqrt(Na*me*1000)*vib
    ## v*n = P*sqrt(mu*mu_to_hartree)*sqrt(1/ma_to_amu)*vib

    Q_P = np.array([Wignerfunc(i, temp) for i in freqs])   # generates update coordinates and momenta pairs Q and P

    Q = Q_P[:, 0].reshape((nfreq, 1))                      # first column is Q

    Q *= 1 / np.sqrt(freqs * mu_to_hartree * ma_to_amu)    # convert coordinates from m to Bohr
    Qvib = np.array([np.ones((natom, 3)) * i for i in Q])  # generate identity array to expand Q
    Qvib = np.sum(vib * Qvib, axis = 0)                    # sum sampled structure over all modes
    newc = (xyz + Qvib) * bohr_to_angstrom                 # cartesian coordinates in Angstrom

    P = Q_P[:, 1].reshape((nfreq, 1))                      # second column is P
    P *= np.sqrt(freqs * mu_to_hartree / ma_to_amu)        # convert velocity from m/s to Bohr/au
    Pvib = np.array([np.ones((natom,3)) * i for i in P])   # generate identity array to expand P
    velo = np.sum(vib * Pvib, axis = 0)                    # sum sampled velocity over all modes in Bohr/au
                                                           # use velo=1 in &DYNAMIX to read un-weighted velocity
    #These lines are for test
    #amass=np.array([np.ones(3)*i for i in amass])         # expand atoic mass over x y z coordinates
    #print(np.linalg.norm(vib[0]*amass**0.5))              # check if mode is mass-weighted (norm=1)  
    #Epot=0.5*np.sum((freqs*mu_to_hartree*Q)**2*ma_to_amu) # The Q has been updated, so it has to be converted back
    #Ekin=0.5*np.sum(amass*ma_to_amu*velo**2)         
    #print(Epot,Ekin,Epot+Ekin)

    #velo*=np.sqrt(amass*ma_to_amu)                        # only use for velo=2 in &DYNAMIX convert velocity[Bohr/au] to mass-weighted [Bohr*amu^0.5/au] for Molcas

    inicond = np.concatenate((newc,velo), axis = 1)
    inicond = np.concatenate((atoms.reshape((-1, 1)), inicond), axis = 1)
    inicond = np.concatenate((inicond, amass), axis = 1)
    inicond = np.concatenate((inicond, achrg), axis = 1)

    return inicond

def Sampling(input, nesmb, iseed, temp, dist, format):
    ## This function recieves input information and does sampling
    ## This function use Readdata to call different functions toextract vibrational frequency and mode
    ## This function calls Boltzmann or Wigner to do sampling
    ## This function returns a list of initial condition 
    ## Import this function for external usage

    if iseed != -1:
        random.seed(iseed)

    callsample = ['molden', 'bagel', 'g16', 'orca']  ## these format need to run sampling
    skipsample = ['newtonx', 'xyz', 'xz']            ## these format read sampled initial conditions

    ## read in function dictionary
    Readdata = {
    'molden' : ReadMolden,
    'bagel'  : ReadBagel,
    'g16'    : ReadG16,
    'orca'   : ReadOrca,
    'newtonx': LoadNewtonX,
    'xyz'    : LoadXYZ,
    'xz'     : LoadXZ,
    }

    if format in callsample:
        sample = Readdata[format](input)
        sample['temp'] = temp
        ensemble = [] # a list of sampled  molecules, ready to transfer to external module or print out

        for s in range(nesmb):
            if   dist == 'boltzmann':
                intcon = Boltzmann(sample)
                ensemble.append(intcon)    
            elif dist == 'wigner':
                intcon = Wigner(sample)
                ensemble.append(intcon)
            sys.stdout.write('Progress: %.2f%%\r' % ((s + 1) * 100 / nesmb))

        q = open('%s-%s-%s.xyz' % (dist, input, temp),'wb')
        p = open('%s-%s-%s.velocity' % (dist, input, temp),'wb')
        pq = open('%s.init' % (input),'wb')
        m = 0
        for mol in ensemble:
            m += 1
            geom = mol[:, 0: 4]   
            velo = mol[:, 4: 7]
            natom = len(geom)
            np.savetxt(
                q,
                geom,
                header = '%s\n [Angstrom]' % (len(geom)),
                comments='',
                fmt = '%-5s%30s%30s%30s')

            np.savetxt(
                p,
                velo,
                header = '%d [Bohr / time_au]' % (m),
                comments='',
                fmt = '%30s%30s%30s')

            np.savetxt(pq,
                mol,
                header = 'Init %5d %5s %12s%30s%30s%30s%30s%30s%22s%6s' % (
                    m,
                    natom,
                    'X(A)',
                    'Y(A)',
                    'Z(A)',
                    'VX(au)',
                    'VY(au)',
                    'VZ(au)',
                    'g/mol',
                    'e'),
                comments = '',
                fmt = '%-5s%30s%30s%30s%30s%30s%30s%16s%6s')
        q.close()
        p.close()
        pq.close()

    elif format in skipsample:
        ensemble = Readdata[format](input)

        if   len(ensemble) < nesmb:
            sys.exit('Not enough initial conditions!!! %s provided < %s requested.' % (len(ensemble),nesmb))
        elif len(ensemble) > nesmb:
            print('More initial conditions received. Skip %s - %s. ' % (nesmb+1,len(ensemble)))
        ensemble = ensemble[0: nesmb]

    return ensemble

def Equilibrium(input, nesmb, iseed, temp, dist, format):
    ## This function recieves input information and read equilibrium geometry
    ## This function use Readdata to call different functions to extract vibrational frequency and mode
    ## This function returns equilibrium geometry
    ## Import this function for external usage

    callsample = ['molden','bagel','g16']  ## these format need to run sampling
    skipsample = ['newtonx','xyz']         ## these format read sampled initial conditions

    ## read in function dictionary
    Readdata = {
    'molden':  ReadMolden,
    'bagel':   ReadBagel,
    'g16':     ReadG16,
    'newtonx': LoadNewtonX,
    'xyz':     LoadXYZ
    }

    if format in callsample:
        sample = Readdata[format](input)
        atoms = sample['atoms']
        xyz = sample['xyz'] * 0.529177249
        amass = sample['amass']
        achrg = sample['achrg']
        eqcond = np.concatenate((xyz, np.zeros(xyz.shape)), axis = 1)
        eqcond = np.concatenate((atoms.reshape((-1, 1)), eqcond), axis = 1)
        eqcond = np.concatenate((eqcond, amass), axis = 1)
        eqcond = np.concatenate((eqcond, achrg), axis = 1)
        return eqcond
    else:
        print('Nothing read from %s' % (input))

def main():
    ## This is the main function 
    ## This function calls Sampling to test the methods.

    usage="""

    Dynamic sampling module for neural network and FSSH

    Usage:
      python3 dynamixsampling.py -i input.freq.molden
      python3 dynamixsampling.py -h for help

    """
    description='Dynamic sampling module for neural network and Molcas TSH'
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-i', dest='input',   type=str,   nargs=1, help='Input freq.molden name.')
    parser.add_option('-n', dest='nesmb',   type=int,   nargs=1, help='Number of structure. Default is 1.',default=1)
    parser.add_option('-s', dest='iseed',   type=int,   nargs=1, help='Random number seed (0 - +inf). Default is random.',default=-1)
    parser.add_option('-t', dest='temp',    type=float, nargs=1, help='Temperature in K. Default is 298.15K.',default=298.15)
    parser.add_option('-d', dest='dist',    type=str,   nargs=1, help='Sampling method: wigner or boltzmann (low-cases). Default is wigner.',default='wigner')

    (options, args) = parser.parse_args()
    if options.input == None:
        print (usage)
        exit()

    input = options.input.split('.')[0]
    format = options.input.split('.')[-1]
    iseed = options.iseed
    temp = options.temp
    dist = options.dist
    nesmb = options.nesmb

    ensemble = Sampling(input, nesmb, iseed, temp, dist, format)[:]
#    n=0
#    for mol in ensemble:
#        n+=1
#        geom=mol[:,0:4]
#       	velo=mol[:,4:]
#               np.savetxt('%s-%d.xyz' % (input,n),geom,header='%s\n ' % (len(geom)),comments='',fmt='%-5s%30s%30s%30s')
#               np.savetxt('%s-%d.velocity.xyz' % (input,n),velo,fmt='%30s%30s%30s')
    
if __name__ == '__main__':
    main()
