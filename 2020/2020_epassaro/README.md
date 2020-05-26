# rad_trans_models

Models used to test spectra uncertainties with different atomic data sets.

Based on Andreas Flörs [scripts and configuration files](https://github.com/tardis-sn/tardis-setups/tree/master/rad_trans_models) and tweaked to work on [MSU ICER HPCC](https://wiki.hpcc.msu.edu/) in the simplest way possible.


## Notes

- `parse_config` function from `util` package allows to parse environment variables from YAML files before to use them in TARDIS simulations. For example, the `ATOM_DATA` variable should be exported beforehand and point to a valid atomic dataset. This workaround avoids keeping lot of almost identical files.

- Pickled objects are saved to the `PICKLE_DIR` variable exported in the `.sb` files.

- `.csvy` files are modified by the Python scripts, so don't try to edit them manually unless you are sure that fields are not overwritten by the script, for example `v_inner_boundary`.

- Merged the two run model functions into a single one with `pickled=True` parameter.

- Removed the multithread `Pool` at the end of the scripts.


## Usage example (SLURM)

### Run

```
sbatch submit.sb run_model_toy06.py
```

### Check status

```
scontrol show job <JOB_ID>
```