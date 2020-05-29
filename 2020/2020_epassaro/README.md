# rad_trans_models

Models used to test spectra uncertainties with different atomic data sets.

Based on Andreas Flörs [scripts and configuration files](https://github.com/tardis-sn/tardis-setups/tree/master/rad_trans_models) and tweaked to work on [MSU ICER HPCC](https://wiki.hpcc.msu.edu/) in the simplest way possible.


## Notes

- `.csvy` files are modified by the Python scripts, so don't try to edit them manually unless you are sure that fields are not overwritten by the script, for example: `v_inner_boundary`.

- Pickled objects are saved to the `PICKLE_DIR` variable exported in the `.sb` files. If not defined, will be saved to the `Output` folder alongside the HDF5 files.

- Merged the two run model functions into a single one with `pickled=True` parameter.

- Removed the multithread `Pool` at the end of the scripts.

- Added a script to sync `Output` folder from MSU ICER HPCC to your local computer.


## Usage example (SLURM)

### Run


```
sbatch submit.sb run_model_<MODEL_NAME>.py <atomic_data_file> 
```

## Custom atomic data files

- Currently we are using the experimental branch `chianti-hdf-new` of Carsus to make the atomic data files (PR [#152](https://github.com/tardis-sn/carsus/pull/152)).

- Remember to download and extract the Chianti Database if you plan to create new atomic data files:

```bash
cd $HOME/Documents
wget http://www.chiantidatabase.org/download/CHIANTI_9.0.1_data.tar.gz
tar -xvz CHIANTI_9.0.1_data.tar.gz -C chianti
export XUVTOP=$HOME/Documents/chianti 
```

- Finally, to run the job on SLURM:

```
sbatch submit.sb create_atom_data.py <SELECTED_IONS>
```


### Check job status

```
scontrol show job <JOB_ID>
```

### Watch log

```
tail -f slurm-<JOB_ID>.out
```