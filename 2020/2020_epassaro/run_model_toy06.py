#!/usr/bin/env python

import os
import sys
import re
import yaml
import pandas as pd
import numpy as np
from tardis.util.base import parse_quantity
from tardis import run_tardis
from astropy import units as u
from tardis.io.config_reader import Configuration
from tardis.simulation import Simulation

pattern_remove_bracket = re.compile('\[.+\]')
t0_pattern = re.compile('tend = (.+)\n')


def read_blondin_toymodel(fname, t_inner=None, w=None):
    with open(fname, 'r') as fh:
        for line in fh:
            if line.startswith("#idx"):
                break
        else:
            raise ValueError(
                'File {0} does not conform to Toy Model format as it does not contain #idx')
    columns = [pattern_remove_bracket.sub(
        '', item) for item in line[1:].split()]

    raw_blondin_csv = pd.read_csv(
        fname, delim_whitespace=True, comment='#', header=None, names=columns)
    raw_blondin_csv.set_index('idx', inplace=True)

    blondin_csv = raw_blondin_csv.loc[:, [
        'vel', 'dens', 'temp', 'X_56Ni0', 'X_Ti', 'X_Ca', 'X_S', 'X_Si', 'X_O', 'X_C']]
    rename_col_dict = {'vel': 'velocity', 'dens': 'density', 'temp': 't_rad'}
    rename_col_dict.update({item: item[2:]
                            for item in blondin_csv.columns[3:]})
    rename_col_dict['X_56Ni0'] = 'Ni56'
    blondin_csv.rename(columns=rename_col_dict, inplace=True)
    blondin_csv.iloc[:, 3:] = blondin_csv.iloc[:, 3:].divide(
        blondin_csv.iloc[:, 3:].sum(axis=1), axis=0)

    # changing velocities to outer boundary
    new_velocities = 0.5 * \
        (blondin_csv.velocity.iloc[:-1].values +
         blondin_csv.velocity.iloc[1:].values)
    new_velocities = np.hstack(
        (new_velocities, [2 * new_velocities[-1] - new_velocities[-2]]))
    blondin_csv['velocity'] = new_velocities

    if t_inner and w:
        blondin_csv['t_inner'] = t_inner
        blondin_csv['w'] = w

    with open(fname, 'r') as fh:
        t0_string = t0_pattern.findall(fh.read())[0]

    t0 = parse_quantity(t0_string.replace('DAYS', 'day'))
    blondin_dict = {}
    blondin_dict['model_density_time_0'] = str(t0)
    blondin_dict['description'] = 'Converted {0} to csvy format'.format(fname)
    blondin_dict['tardis_model_config_version'] = 'v1.0'
    blondin_dict_fields = [
        dict(name='velocity', unit='km/s', desc='velocities of shell outer bounderies.')]
    blondin_dict_fields.append(
        dict(name='density', unit='g/cm^3', desc='mean density of shell.'))
    blondin_dict_fields.append(
        dict(name='t_rad', unit='K', desc='radiative temperature.'))

    if t_inner and w:
        blondin_dict_fields.append(
            dict(name='t_inner', unit='K', desc='inner boundary temperature.'))
        blondin_dict_fields.append(
            dict(name='w', desc='dilution factor.'))

    for abund in blondin_csv.columns[3:]:
        blondin_dict_fields.append(
            dict(name=abund, desc='Fraction {0} abundance'.format(abund)))
    blondin_dict['datatype'] = {'fields': blondin_dict_fields}

    return blondin_dict, blondin_csv


def run_tardis_model(params):
    model_config = Configuration.from_yaml('blondin_model_compare_06.yml')
    model_config.model.v_inner_boundary = params[2]
    model_config.model.v_outer_boundary = 35000*u.km/u.s
    model_config.supernova.luminosity_requested = params[1]
    model_config.supernova.time_explosion = params[0]
    model_config.atom_data = sys.argv[1]

    # Run only one iteration for Chianti models
    if 'chianti' in model_config.atom_data:
        model_config.montecarlo.iterations = 1

    sim = Simulation.from_config(model_config)
    sim.run()

    atom_dir = model_config.atom_data.strip('.h5')
    full_path = os.path.join('Output', 'Toy_06', atom_dir)
    os.makedirs(full_path, exist_ok=True)

    fname = '{}/toy06_t{}_v{}.h5'.format(
        full_path, params[0].value, params[2].value)

    with pd.HDFStore(fname) as hdf:
        hdf.put('wavelength', pd.Series(
            sim.runner.spectrum.wavelength.value))
        hdf.put('luminosity_density_lambda', pd.Series(
            sim.runner.spectrum_integrated.luminosity_density_lambda.value))
        hdf.put('w', pd.Series(sim.plasma.w))
        hdf.put('t_electrons', pd.Series(sim.plasma.t_electrons))
        hdf.put('ion_number_density', sim.plasma.ion_number_density)
        hdf.put('electron_densities', sim.plasma.electron_densities)
        hdf.put('t_rad', pd.Series(sim.plasma.t_rad))
        hdf.put('r_inner_cgs', pd.Series(sim.runner.r_inner_cgs))
        hdf.put('t_inner', pd.Series(sim.iterations_t_inner))

    return 1


try:
    t_inner = pd.read_hdf(sys.argv[2], 't_inner')
    w = pd.read_hdf(sys.argv[2], 'w')
    blondin_dict, blondin_csv = read_blondin_toymodel(
        'snia_toy06.dat', t_inner, w)

except IndexError:
    blondin_dict, blondin_csv = read_blondin_toymodel('snia_toy06.dat')

blondin_dict['v_inner_boundary'] = '9000 km/s'
blondin_dict['v_outer_boundary'] = '35000 km/s'
blondin_dict['model_isotope_time_0'] = '0. d'
csvy_file = '---\n{0}\n---\n{1}'.format(yaml.dump(
    blondin_dict, default_flow_style=False),
    blondin_csv.to_csv(index=False))

# Create CSVY file
with open('blondin_compare_06.csvy', 'w') as fh:
    fh.write(csvy_file)

lbols = np.array([3.05e+42, 8.91e+42, 1.10e+43, 1.00e+43])*u.erg/u.s

# final_params = [(5*u.d, lbols[0],  20500.*u.km/u.s),
#                 (10*u.d, lbols[1], 17000.*u.km/u.s),
#                 (15*u.d, lbols[2], 10000.*u.km/u.s),
#                 (20*u.d, lbols[3], 5500.*u.km/u.s)]

final_params = [(15*u.d, lbols[2], 10000.*u.km/u.s)]

for params in final_params:
    run_tardis_model(params)

sys.exit(0)
