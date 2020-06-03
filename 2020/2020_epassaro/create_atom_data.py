#! /usr/bin/env python3

import os
import sys
from carsus.io.nist import NISTWeightsComp, NISTIonizationEnergies
from carsus.io.kurucz import GFALLReader
from carsus.io.zeta import KnoxLongZeta
from carsus.io.chianti_ import ChiantiReader
from carsus.io.output import TARDISAtomData


CARSUS_OUT = os.environ['CARSUS_OUT']
GFALL_IONS = 'H-Zn'

try:
    chianti_ions = sys.argv[1]

except IndexError:
    chianti_ions = None

parsers = []
parsers.append(NISTWeightsComp(GFALL_IONS))
parsers.append(NISTIonizationEnergies(GFALL_IONS))
parsers.append(KnoxLongZeta())
parsers.append(GFALLReader(ions=GFALL_IONS))

if chianti_ions:
    parsers.append(ChiantiReader(ions=chianti_ions, priority=20))
    ions_string = chianti_ions.replace(' ', '_').replace(';', '')
    output_file = 'kurucz_cd23_latest_chianti_{}.h5'.format(ions_string)
else:
    output_file = 'kurucz_cd23_latest.h5'

os.makedirs(CARSUS_OUT, exist_ok=True)
fname = os.path.join(CARSUS_OUT, output_file)
atom_data = TARDISAtomData(*parsers)
atom_data.to_hdf(fname)


sys.exit(0)
