"""
Tardis 2020 Example
==========================

This is the example output 30d
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

conf = Configuration.from_yaml(
    "../../2020/2020_williamson_94I/code_comp_setups/30d.yml"
)
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
