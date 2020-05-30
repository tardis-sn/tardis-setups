#! /usr/bin/env python3

import os
import sys
import carsus
from carsus.io.nist import NISTWeightsComp, NISTIonizationEnergies
from carsus.io.kurucz import GFALLReader
from carsus.io.zeta import KnoxLongZeta
from carsus.io.chianti_ import ChiantiReader
from carsus.io.output import TARDISAtomData

CARSUS_DIR = os.path.dirname(carsus.__file__)
CARSUS_OUT = os.environ['CARSUS_OUT']

gfall_ions = 'H-Zn'
gfall_url = 'http://kurucz.harvard.edu/linelists/gfall/gfall.dat'
os.system('wget -q -O /tmp/gfall.dat {}'.format(gfall_url))

try:
    chianti_ions = sys.argv[1]
except IndexError:
    chianti_ions = None

parsers = []
parsers.append(NISTWeightsComp(gfall_ions))
parsers.append(NISTIonizationEnergies(gfall_ions))
parsers.append(GFALLReader('/tmp/gfall.dat', gfall_ions))
parsers.append(KnoxLongZeta(os.path.join(CARSUS_DIR,
                                         'data/knox_long_recombination_zeta.dat')))

if chianti_ions:
    parsers.append(ChiantiReader(chianti_ions, priority=20))
    ions_string = chianti_ions.replace(' ', '_').replace(';', '')
    output_file = 'kurucz_cd23_latest_chianti_{}.h5'.format(ions_string)
else:
    output_file = 'kurucz_cd23_latest.h5'

os.makedirs(CARSUS_OUT, exist_ok=True)
fname = os.path.join(CARSUS_OUT, output_file)
atom_data = TARDISAtomData(*parsers)
atom_data.to_hdf(fname)


sys.exit(0)
