"""
Tardis 2014 Example
==========================

This is the example
"""


from tardis import run_tardis
from tardis.io.config_reader import Configuration
from tardis.io.atom_data.util import download_atom_data
import matplotlib.pyplot as plt

# %%
# Uncomment this line if you need to download the dataset.


download_atom_data("kurucz_cd23_chianti_H_He")

# %%
# Runs the example
def config_modifier(conf):
    conf["montecarlo"]["nthreads"] = 1
    conf["montecarlo"]["last_no_of_packets"] = 1.0e5
    conf["montecarlo"]["no_of_virtual_packets"] = 10

    conf["atom_data"] = "/home/blackbird/Downloads/tardis-data/" + conf["atom_data"]

    return conf


conf = Configuration.from_yaml(
    "../../2014/2014_kerzendorf_sim/appendix_A1/tardis_example.yml"
)
modified_conf = config_modifier(conf)

print(modified_conf)

# %%
sim = run_tardis(modified_conf)


# %%
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
