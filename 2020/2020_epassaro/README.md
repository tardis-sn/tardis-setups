# rad_trans_models

Models used to test spectra uncertainties with different atomic data sets.

Based on Andreas Flörs [scripts and configuration files](https://github.com/tardis-sn/tardis-setups/tree/master/rad_trans_models) and tweaked to work on [MSU ICER HPCC](https://wiki.hpcc.msu.edu/) in the simplest way possible.


## Notes

- `atom_data` from YAML configuration file will be replaced by the `ATOM_DATA` environment variable. 

- `.csvy` files are modified by the Python scripts, so don't try to edit them manually unless you are sure that fields are not overwritten by the script, for example `v_inner_boundary`.

- Pickled objects are saved to the `PICKLE_DIR` variable exported in the `.sb` files. If not defined, will be saved to the `Output` folder alongside the HDF5 files.

- Merged the two run model functions into a single one with `pickled=True` parameter.

- Removed the multithread `Pool` at the end of the scripts.

- Added a script to sync `Output` folder from MSU ICER HPCC to your local computer.

## Usage example (SLURM)

### Run

```
sbatch submit.sb run_model_toy06.py
```

### Check status

```
scontrol show job <JOB_ID>
```

### Watch logfile

```
tail -f slurm-<JOB_ID>.out
```