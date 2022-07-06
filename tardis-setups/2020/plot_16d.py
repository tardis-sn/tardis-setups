"""
SN 1994I spectra 16 days post Explosion
==========================

The following result is obtained using the configuration file from the paper
Williamson, Marc, Kerzendorf, Wolfgang, Modjaz, Maryam 2021, ApJ,
“Modeling Type Ic Supernovae with TARDIS: Hidden Helium in SN 1994I?” (`ADS Link`_).
This plot shows the output obtained by TARDIS for the SN 1994I spectra 16 days post explosion.
Please refer figure 2 from the paper to compare the results of the paper.

.._ADS Link: https://ui.adsabs.harvard.edu/abs/2021ApJ...908..150W
"""


from tardis import run_tardis
from tardis.io.config_reader import Configuration
from tardis.io.atom_data.util import download_atom_data
import matplotlib.pyplot as plt

import sys

sys.path.append("../")
from setup_utils import config_modifier

# %%
# Comment this line if you do not need to download the dataset.


download_atom_data("kurucz_cd23_chianti_H_He")

# %%
conf = Configuration.from_yaml(
    "../../2020/2020_williamson_94I/code_comp_setups/16d.yml"
)

# %%
# Note: Here the configuration is slightly modified to allow
# the configuration file on a computer with lower configuration.

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
