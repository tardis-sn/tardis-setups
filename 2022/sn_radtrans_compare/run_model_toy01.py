import pandas as pd
import re
import numpy as np
import yaml
from tardis.util.base import parse_quantity
from tardis import run_tardis
from astropy import units as u
import base64
import numpy as np
from tardis.io.config_reader import Configuration
from tardis.simulation import Simulation
from multiprocessing import Pool


pattern_remove_bracket = re.compile('\[.+\]')
t0_pattern = re.compile('tend = (.+)\n')


def read_blondin_toymodel(fname):
    with open(fname, 'r') as fh:
        for line in fh:
            if line.startswith("#idx"):
                break
        else:
            raise ValueError('File {0} does not conform to Toy Model format as it does not contain #idx')
    columns = [pattern_remove_bracket.sub('', item) for item in line[1:].split()]

    raw_blondin_csv =  pd.read_csv(fname, delim_whitespace=True, comment='#', header=None, names=columns)
    raw_blondin_csv.set_index('idx', inplace=True)

    blondin_csv = raw_blondin_csv.loc[:, ['vel', 'dens', 'temp', 'X_56Ni0', 'X_Ti', 'X_Ca', 'X_S', 'X_Si', 'X_O', 'X_C']]
    rename_col_dict = {
        'vel': 'velocity',
        'dens': 'density',
        'temp': 't_rad',
    } | {item: item[2:] for item in blondin_csv.columns[3:]}
    rename_col_dict['X_56Ni0'] = 'Ni56'
    blondin_csv.rename(columns=rename_col_dict, inplace=True)
    blondin_csv.iloc[:, 3:] = blondin_csv.iloc[:,3:].divide(blondin_csv.iloc[:,3:].sum(axis=1), axis=0)


    #changing velocities to outer boundary
    new_velocities = 0.5 * (blondin_csv.velocity.iloc[:-1].values + blondin_csv.velocity.iloc[1:].values)
    new_velocities = np.hstack((new_velocities, [2 * new_velocities[-1] - new_velocities[-2]]))
    blondin_csv['velocity'] = new_velocities



    with open(fname, 'r') as fh:
        t0_string = t0_pattern.findall(fh.read())[0]

    t0 = parse_quantity(t0_string.replace('DAYS', 'day'))
    blondin_dict_fields = [
        dict(
            name='velocity',
            unit='km/s',
            desc='velocities of shell outer bounderies.',
        ),
        dict(name='density', unit='g/cm^3', desc='mean density of shell.'),
        dict(name='t_rad', unit='K', desc='radiative temperature.'),
    ]
    blondin_dict_fields.extend(
        dict(name=abund, desc='Fraction {0} abundance'.format(abund))
        for abund in blondin_csv.columns[3:]
    )
    blondin_dict = {
        'model_density_time_0': str(t0),
        'description': 'Converted {0} to csvy format'.format(fname),
        'tardis_model_config_version': 'v1.0',
        'datatype': {'fields': blondin_dict_fields},
    }
    return blondin_dict, blondin_csv

blondin_dict, blondin_csv = read_blondin_toymodel('snia_toy01.dat')

blondin_dict['v_inner_boundary'] = '9000 km/s'
blondin_dict['v_outer_boundary'] = '35000 km/s'
blondin_dict['model_isotope_time_0'] = '0. d'

csvy_file = '---\n{0}\n---\n{1}'.format(yaml.dump(blondin_dict, default_flow_style=False), blondin_csv.to_csv(index=False))

with open('blondin_compare_01.csvy', 'w') as fh:
    fh.write(csvy_file)
epochs = np.array([5,10,15,20])*u.d
velocity_grid = np.arange(10000,26500, 500)*u.km/u.s

lbols = np.array([1.93e+41, 1.68e+42, 2.19e+42, 1.92e+42])*u.erg/u.s


model_grid = np.array(np.empty((epochs.shape[0]*velocity_grid.shape[0], 3)))
model_grid = []
for i, epoch in enumerate(epochs):
    model_grid.extend(
        (epoch, lbols[i], velocity - 2000 * u.km / u.s * i)
        for velocity in velocity_grid
    )
print(len(model_grid))


def run_tardis_model(params):
    model_config = Configuration.from_yaml('blondin_model_compare_01.yml')
    model_config.model.v_inner_boundary = params[2]
    model_config.model.v_outer_boundary = 35000*u.km/u.s
    model_config.supernova.luminosity_requested = params[1]
    model_config.supernova.time_explosion = params[0]
    sim = Simulation.from_config(model_config)
    print(sim.model.v_boundary_inner)
    sim.run()
    fname = f'Output/Toy_01/toy01_t{params[0].value}_v{params[2].value}.hdf'
    with pd.HDFStore(fname) as hdf:
        hdf.put('wavelength', pd.Series(sim.runner.spectrum.wavelength.value))
        hdf.put('lum', pd.Series(sim.runner.spectrum_integrated.luminosity_density_lambda.value))
        hdf.put('w', pd.Series(sim.plasma.w))
        hdf.put('t_electrons', pd.Series(sim.plasma.t_electrons))
        hdf.put('ion_num_dens', sim.plasma.ion_number_density)
        hdf.put('electron_dens', sim.plasma.electron_densities)
    return 1

def run_final_models_plus_pickle(params, fname='blondin_model_compare_01.yml'):
    model_config = Configuration.from_yaml(fname)
    model_config.model.v_inner_boundary = params[2]
    model_config.model.v_outer_boundary = 35000*u.km/u.s
    model_config.supernova.luminosity_requested = params[1]
    model_config.supernova.time_explosion = params[0]
    sim = Simulation.from_config(model_config)
    print(sim.model.v_boundary_inner)
    sim.run()
    import pickle
    dump = f'Output/Toy_01/toy01_t{params[0].value}_v{params[2].value}.pickle'
    with open(dump, 'wb') as dumpfile:
        pickle.dump(sim, dumpfile)
    return 1

pool = Pool(4)
final_params = [(5*u.d, lbols[0],  19000.*u.km/u.s), (10*u.d, lbols[1], 14000.*u.km/u.s), (15*u.d, lbols[2], 11000.*u.km/u.s), (20*u.d, lbols[3], 9000.*u.km/u.s)]
pool.map(run_final_models_plus_pickle, final_params)
