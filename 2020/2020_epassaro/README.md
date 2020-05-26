# rad_trans_models

Models used to test spectra uncertainties with different atomic data sets.

Based on Andreas Flörs [scripts and configuration files](https://github.com/tardis-sn/tardis-setups/tree/master/rad_trans_models) and tweaked to work on [MSU ICER HPCC](https://wiki.hpcc.msu.edu/).


## Notes

- `parse_config` function from `util` package allows to parse environment variables from YAML files before to use them in TARDIS simulations. For example, the `ATOM_DATA` variable should be exported beforehand and point to a valid atomic dataset. This workaround avoids keeping lot of almost identical files.

- `.csvy` files are modified by the Python scripts, so don't try to edit them manually unless you are sure that fields are not overwritten by the script.

- Pickled objects are saved to the `PICKLE_DIR` variable exported in the `.sb` files.