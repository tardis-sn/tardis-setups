from tardis import run_tardis
from tardis.io.atom_data.util import download_atom_data
import matplotlib.pyplot as plt

# download_atom_data("kurucz_cd23_chianti_H_He")

sim = run_tardis(
    "/home/blackbird/Rohith/tardis/tardis-setups/2020/2020_williamson_94I/code_comp_setups/22d.yml"
)

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
