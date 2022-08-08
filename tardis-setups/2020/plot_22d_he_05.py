"""
Williamson et al. (2020) Figure 3 g He=0.05M⊙
==========================


Article: Williamson, Marc, Kerzendorf, Wolfgang, Modjaz, Maryam 2021, ApJ,
“Modeling Type Ic Supernovae with TARDIS: Hidden Helium in SN 1994I?” (`ADS Link`_).

Original Input Files: `YAML`_, `CSVY`_

Original Atomic Dataset: Data missing

Original Spectra: Data missing

Notes: Please note that the spectra obtained below is obtained by using a slightly
modified configuration file. This is done to ensure that the spectra can be
obtained using the computers hosted by us.

.. _ADS Link: https://ui.adsabs.harvard.edu/abs/2021ApJ...908..150W
.. _YAML: https://github.com/tardis-sn/tardis-setups/blob/master/2020/2020_williamson_94I/he_setups/22d_he_05.yml
.. _CSVY: https://github.com/tardis-sn/tardis-setups/blob/master/2020/2020_williamson_94I/he_setups/22d_He_0.05.csvy
"""


from tardis import run_tardis
from tardis.io.config_reader import Configuration
from tardis.io.atom_data.util import download_atom_data
import matplotlib.pyplot as plt

import sys

sys.path.append("../")
from setup_utils import config_modifier

# %%
# Uncomment this line if you need to download the dataset.


download_atom_data("kurucz_cd23_chianti_H_He")

# %%
# Runs the example

conf = Configuration.from_yaml("../../2020/2020_williamson_94I/he_setups/22d_he_05.yml")
conf = config_modifier(conf)
# %%
sim = run_tardis(conf)


spectrum = sim.runner.spectrum
spectrum_virtual = sim.runner.spectrum_virtual
spectrum_integrated = sim.runner.spectrum_integrated

plt.figure(figsize=(10, 6.5))

spectrum.plot(label="Normal packets")
spectrum_virtual.plot(label="Virtual packets")
spectrum_integrated.plot(label="Formal integral")

plt.xlim(500, 9000)
plt.title("TARDIS example model spectrum")
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel("Luminosity density [erg/s/$\AA$]")
plt.legend()
plt.show()
