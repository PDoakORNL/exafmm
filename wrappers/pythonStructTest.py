import sys
import re
import os
import numpy as np

sys.path.append("/Users/epd/codes/electrostatics")

import ase.io.xyz
import ase.io
import ase.constraints
from ase.io.bader import attach_charges

from ase.calculators.siesta import Siesta
from ase.data.vdw import vdw_radii as vdw_radii

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import minimize# as fmin_tnc
from scipy.optimize import basinhopping# as fmin_tnc

import mesoelectrostatics

from islandoptimizer import IslandOptimizer

import math
import cStringIO


import argparse
from ParseListAction import ParseListAction
parser = argparse.ArgumentParser()
parser.add_argument("-a","--acceptor", dest="aPath",
                    help="acceptor path",
                    default="/Users/epd/projects/TTF-TCNQ/charges/Siesta/tcnq_symmetrized")
parser.add_argument("-d","--donor", dest="dPath",
                    help="donor path",
                    default="/Users/epd/projects/TTF-TCNQ/charges/Siesta/ttf_symmetrized")
parser.add_argument("-u","--up", dest="thick_up",
                    help="is the thick edge up?", default=None, action="store_true")
parser.add_argument("--sacceptor", dest="aStem",
                    help="acceptor stem",
                    default="TCNQ")
parser.add_argument("-c","--chargeType", dest="chargeType",
                    help="Charge type to read",
                    default="Voronoi")
parser.add_argument("--sdonor", dest="dStem",
                    help="donor stem",
                    default="TTF")
parser.add_argument("--initialGuess", dest="initial_guess",
                    help="initial guess [yadda,yadda,yadda]",
                    action=ParseListAction, default=[0.00,0.00,0.00])
parser.add_argument("-l","--lattice", dest="lattice",
                    help="island bulk region size exp [4,2]",
                    action=ParseListAction, default=[4,2])
parser.add_argument("-t", "--transferedCharge", dest="charge_transfer",
                    help="chargeTranser",
                    default=0.00)
parser.add_argument("-e","--eps", dest="eps",
                    help="epsilon",
                    default=0.001)
parser.add_argument("-o","--out", dest="out",
                    help="output Label",
                    default="optimize/TTF")
parser.add_argument("-b","--bounds", dest="bounds",
                    help="initial guess [yadda,yadda,yadda]",
                    default=[[-0.2,0.0],[-0.14,0.14],[-np.pi/2, np.pi/2]])

options = parser.parse_args()

islandOpt = IslandOptimizer(planarize=True)
the1x1, centerTTF, centerTCNQ, ttfAngle, tcnqAngle = islandOpt.basic_cell()
energyOut = open("{}.eng".format(options.out), 'w')

#this is a bit hacky and assumes the charge series structure for a siesta run
# as also appears in one point edge charges
charge = float(options.charge_transfer)
structTemp = "{}/{}_0.00_c{:3.2f}/{}_0.00_c{:3.2f}.STRUCT_IN"
acharge = 0.0
if math.fabs(charge) > 0.00001:
    acharge = -charge

aStruct = structTemp.format(options.aPath,options.aStem,acharge,
                            options.aStem, acharge)
print aStruct
islandOpt.read_acceptor(aStruct)
dStruct = structTemp.format(options.dPath,options.dStem,charge,
                            options.dStem, charge)
print dStruct
islandOpt.read_donor(dStruct)

outTemp = "{}/{}_0.00_c{:3.2f}/{}_opt_0.00_c{:3.2f}.out"
aOut = outTemp.format(options.aPath,options.aStem,acharge,
                            options.aStem, acharge)
print outTemp
tcnqSymEq=[[16,17,18,19],[1,3,6,10],[0,2],[9,11],[12,13,14,15],[4,5,7,8]]
islandOpt.read_acceptor_charges(aOut,chargeType=options.chargeType, symmetries=tcnqSymEq)

dOut = outTemp.format(options.dPath,options.dStem,charge,
                            options.dStem, charge)
print outTemp
ttfSymEq=[[6,7,8,9],[1,2,4,5],[10,11,12,13],[0,3]]
islandOpt.read_donor_charges(dOut,chargeType=options.chargeType, symmetries=ttfSymEq)

myisland = islandOpt.getIsland(options.initial_guess[0],
                               options.initial_guess[1],
                               options.initial_guess[2],
                               options.thick_up,
                               options.lattice)

#myisland.translate(np.array([[100,100,100]]*len(myisland.positions)))
print myisland.positions
ase.io.write("{}.eps".format(options.out),myisland,show_tags=True)

from ctypes import *
import numpy as np
import random
py_mes = CDLL("libpythonMesSer.so")

Nmax = c_int(len(myisland));
Ni = c_int(len(myisland));
stringLength = c_int(20);
images = c_int(0);
ksize = c_int(11);
cycle = c_double(2 * np.pi);
alpha = c_double(10 / cycle.value)
sigma = c_double(.25 / np.pi)
cutoff = c_double(cycle.value * alpha.value / 3)
NmaxInt = c_int * Nmax.value
index = NmaxInt()
#int * index = new int [Nmax];
NmaxDouble = c_double * Nmax.value
ThreeNmaxDouble = c_double * (Nmax.value * 3)
x = ThreeNmaxDouble()
q = NmaxDouble()
p = NmaxDouble()
f = ThreeNmaxDouble()
p2 = NmaxDouble()
f2 = ThreeNmaxDouble()

for i,atom in enumerate(myisland):
    for j in range(3):
        x[3*i+j]=atom.position[j]
    q[i]=atom.charge

average = 0

py_mes.FMM_Init(images)
print "done init"

fmm_partition = py_mes.FMM_Partition_NonP
print fmm_partition
fmm_partition.argtypes = [POINTER(c_int),
                          POINTER(c_int),
                          POINTER(c_double),
                          POINTER(c_double)]

#print "Ni:", Ni, "index:", index, "x", x, "q", q
fmm_partition(Ni, index, x, q)
print "done partition"

molSums = []
molSumsPLJ = []

for thisTag in set(myisland.get_tags()):
    print "At tag", thisTag
    targetAtoms = [index for index,tag in enumerate(myisland.get_tags()) if tag==thisTag]
    sourceAtoms = [index for index,tag in enumerate(myisland.get_tags()) if tag!=thisTag]

    print "pre sort"
    xTarg = np.array([myisland.positions[i] for i in targetAtoms]).flatten().tolist()
    xSource = np.array([myisland.positions[i] for i in sourceAtoms]).flatten().tolist()
    qTarg = np.array([myisland[i].charge for i in targetAtoms]).flatten().tolist()
    qSource = np.array([myisland[i].charge for i in sourceAtoms]).flatten().tolist()
    ljTarg = np.array([vdw_radii[myisland[i].number] for i in targetAtoms]).flatten().tolist()
    ljSource = np.array([vdw_radii[myisland[i].number] for i in sourceAtoms]).flatten().tolist()
    
    Nta = (len(targetAtoms))
    Nsa = (len(sourceAtoms))
    TargNmaxDouble = c_double * Nta
    TargNmax3Double = c_double * (Nta * 3)
    SourceNmaxDouble = c_double * Nsa
    SourceNmax3Double = c_double * (Nsa * 3)

    pxTarg = TargNmax3Double()
    pqTarg = TargNmaxDouble()
    pxSource = SourceNmax3Double()
    pqSource = SourceNmaxDouble()
    p2 = TargNmaxDouble()
    pljTarg = SourceNmaxDouble()
    pljSource = SourceNmaxDouble()
    plj = TargNmaxDouble()
    pes = TargNmaxDouble()

    for i in range(Nta):
        p2[i] = 0
        plj[i] = 0
        pes[i] = 0

    for i in range(Nta):
        for j in range(3):
            pxTarg[3*i+j] = xTarg[3*i+j]
        pqTarg[i] = qTarg[i] 
        pljTarg[i] = ljTarg[i]
    
    for i in range(Nsa):
        for j in range(3):
            pxSource[3*i+j] = xSource[3*i+j]
        pqSource[i] = qSource[i] 
        pljSource[i] = ljSource[i]
        
    direct_coulomb_ts = py_mes.Direct_Coulomb_TS
    direct_coulomb_ts.argtypes = [POINTER(c_int),
                                  POINTER(c_int),
                                  POINTER(c_double),
                                  POINTER(c_double),
                                  POINTER(c_double),
                                  POINTER(c_double),
                                  POINTER(c_double)]
    
    direct_coulomb_ts(pointer(c_int(Nta)),
                      pointer(c_int(Nsa)),
                      pxTarg,
                      pxSource,
                      pqTarg,
                      pqSource,
                      p2)

    molSums.append([thisTag,sum(p2)])

    direct_coulomb_lj_ts = py_mes.Direct_Coulomb_LJ_TS

    direct_coulomb_lj_ts.argtypes = [POINTER(c_int),
                                     POINTER(c_int),
                                     POINTER(c_double),
                                     POINTER(c_double),
                                     POINTER(c_double),
                                     POINTER(c_double),
                                     POINTER(c_double),
                                     POINTER(c_double),
                                     POINTER(c_double),
                                     POINTER(c_double),
                                     c_double]
    
    
    direct_coulomb_lj_ts(pointer(c_int(Nta)),
                         pointer(c_int(Nsa)),
                         pxTarg,
                         pxSource,
                         pqTarg,
                         pqSource,
                         pljTarg,
                         pljSource,
                         pes,
                         plj,
                         c_double(options.eps))

    molSumsPLJ.append([thisTag,(sum(plj)+sum(pes))])

    
for molSum in molSums:
    print "tag:", molSum[0], "energy:", molSum[1]
for molSum in molSumsPLJ:
    print "tag:", molSum[0], "PLJenergy:", molSum[1]

    
py_mes.FMM_Finalize()
#py_mes.mpi_finalize()


#would like to print final island structure
