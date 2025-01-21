import shutil
import subprocess
import tempfile
from filecmp import dircmp
from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode
import random
import plotly.colors as pc
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import logging

init_notebook_mode()

logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'compare_path': '.', 
    'temp_dir_prefix': 'ref_compare_',
}

# Utility functions
def color_print(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def get_relative_path(path, base):
    return str(Path(path).relative_to(base))

def get_last_two_commits():
    try:
        result = subprocess.run(['git', 'log', '--format=%H', '-n', '2'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        commits = result.stdout.strip().split('\n')
        if len(commits) >= 2:
            return commits[1], commits[0]
        return None, None
    except (subprocess.SubprocessError, subprocess.CalledProcessError):
        print("Error: Unable to get git commits.")
        return None, None

class FileManager:
    def __init__(self):
        self.temp_dir = None

    def setup(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix=CONFIG['temp_dir_prefix']))
        print(f'Created temporary directory at {self.temp_dir}')

    def teardown(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f'Removed temporary directory {self.temp_dir}')
        self.temp_dir = None

    def get_temp_path(self, filename):
        return str(self.temp_dir / filename)

    def copy_file(self, source, destination):
        shutil.copy2(source, self.get_temp_path(destination))

class FileSetup:
    def __init__(self, file_manager, ref1_hash, ref2_hash):
        self.file_manager = file_manager
        self.ref1_hash = ref1_hash
        self.ref2_hash = ref2_hash

    def setup(self):
        for ref_id, ref_hash in enumerate([self.ref1_hash, self.ref2_hash], 1):
            ref_dir = self.file_manager.get_temp_path(f"ref{ref_id}")
            os.makedirs(ref_dir, exist_ok=True)
            if ref_hash:
                self._copy_data_from_hash(ref_hash, ref_dir)
            else:
                subprocess.run(f'cp -r {CONFIG["compare_path"]}/* {ref_dir}', shell=True)

    def _copy_data_from_hash(self, ref_hash, ref_dir):
        git_cmd = ['git', 'archive', ref_hash, '|', 'tar', '-x', '-C', str(ref_dir)]
        subprocess.run(' '.join(git_cmd), shell=True)

class DiffAnalyzer:
    def __init__(self, file_manager):
        self.file_manager = file_manager

    def display_diff_tree(self, dcmp, prefix=''):
        for item in sorted(dcmp.left_only):
            path = Path(dcmp.left) / item
            self._print_item(f'{prefix}−', item, 'red', path.is_dir())

        for item in sorted(dcmp.right_only):
            path = Path(dcmp.right) / item
            self._print_item(f'{prefix}+', item, 'green', path.is_dir())

        for item in sorted(dcmp.diff_files):
            self._print_item(f'{prefix}✱', item, 'yellow')

        for item in sorted(dcmp.common_dirs):
            self._print_item(f'{prefix}├', item, 'blue', True)
            subdir = getattr(dcmp, 'subdirs')[item]
            self.display_diff_tree(subdir, prefix + '│ ')

    def _print_item(self, symbol, item, color, is_dir=False):
        dir_symbol = '/' if is_dir else ''
        color_print(f"{symbol} {item}{dir_symbol}", color)

    def print_diff_files(self, dcmp):
        dcmp.right = Path(dcmp.right)
        dcmp.left = Path(dcmp.left)
        
        self._print_new_files(dcmp.right_only, dcmp.right, "ref1")
        self._print_new_files(dcmp.left_only, dcmp.left, "ref2")
        self._print_modified_files(dcmp)

        for sub_dcmp in dcmp.subdirs.values():
            self.print_diff_files(sub_dcmp)

    def _print_new_files(self, files, path, ref):
        for item in files:
            if Path(path, item).is_file():
                print(f"New file detected inside {ref}: {item}")
                print(f"Path: {Path(path, item)}")
                print()

    def _print_modified_files(self, dcmp):
        for name in dcmp.diff_files:
            print(f"Modified file found {name}")
            left = self._get_relative_path(dcmp.left)
            right = self._get_relative_path(dcmp.right)
            if left == right:
                print(f"Path: {left}")
            print()

    def _get_relative_path(self, path):
        try:
            return str(Path(path).relative_to(self.file_manager.temp_dir))
        except ValueError:
            # If the path is not relative to temp_dir, return the full path
            return str(path)

class HDFComparator:
    def __init__(self, print_path=False):
        self.print_path = print_path

    def summarise_changes_hdf(self, name, path1, path2):
        ref1 = pd.HDFStore(Path(path1) / name)
        ref2 = pd.HDFStore(Path(path2) / name)
        k1, k2 = set(ref1.keys()), set(ref2.keys())
        
        different_keys = len(k1 ^ k2)
        identical_items = []
        identical_name_different_data = []
        identical_name_different_data_dfs = {}

        for item in k1 & k2:
            try:
                if ref1[item].equals(ref2[item]):
                    identical_items.append(item)
                else:
                    identical_name_different_data.append(item)
                    identical_name_different_data_dfs[item] = (ref1[item] - ref2[item]) / ref1[item]
                    self._compare_and_display_differences(ref1[item], ref2[item], item, name, path1, path2)
            except Exception as e:
                print(f"Error comparing item: {item}")
                print(e)

        ref1.close()
        ref2.close()

        # Only return results if there are differences
        if different_keys > 0 or len(identical_name_different_data) > 0:
            print("\n" + "=" * 50)  # Add a separator line
            print(f"Summary for {name}:")
            print(f"Total number of keys- in ref1: {len(k1)}, in ref2: {len(k2)}")
            print(f"Number of keys with different names in ref1 and ref2: {different_keys}")
            print(f"Number of keys with same name but different data in ref1 and ref2: {len(identical_name_different_data)}")
            print(f"Number of totally same keys: {len(identical_items)}")
            print("=" * 50)  # Add another separator line after the summary
            print()

        return {
            "different_keys": different_keys,
            "identical_keys": len(identical_items),
            "identical_keys_diff_data": len(identical_name_different_data),
            "identical_name_different_data_dfs": identical_name_different_data_dfs,
            "ref1_keys": list(k1),
            "ref2_keys": list(k2)
        }

    def _compare_and_display_differences(self, df1, df2, item, name, path1, path2):
        abs_diff = np.fabs(df1 - df2)
        rel_diff = abs_diff / np.maximum(np.fabs(df1), np.fabs(df2))

        # Check for differences larger than floating point uncertainty
        FLOAT_UNCERTAINTY = 1e-14
        max_rel_diff = np.nanmax(rel_diff)  # Using nanmax to handle NaN values

        if max_rel_diff > FLOAT_UNCERTAINTY:
            logger.warning(
                f"Significant difference detected in {name}, key={item}\n"
                f"Maximum relative difference: {max_rel_diff:.2e} "
                f"(Versions differ by {max_rel_diff*100:.2e}%)"
            )

        print(f"Displaying heatmap for key {item} in file {name} \r")
        for diff_type, diff in zip(["abs", "rel"], [abs_diff, rel_diff]):
            print(f"Visualising {'Absolute' if diff_type == 'abs' else 'Relative'} Differences")
            self._display_difference(diff)

        if self.print_path:
            if path1 != path2:
                print(f"Path1: {path1}")
                print(f"Path2: {path2}")
            else:
                print(f"Path: {path1}")


    def _display_difference(self, diff):
        with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
            if isinstance(diff, pd.Series):
                diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
            elif isinstance(diff.index, pd.core.indexes.multi.MultiIndex):
                diff = diff.reset_index(drop=True)
            
            diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
            display(diff.style.format('{:.2g}'.format).background_gradient(cmap='Reds'))


class SpectrumSolverComparator:
    def __init__(self, ref1_path, ref2_path):
        self.ref1_path = ref1_path
        self.ref2_path = ref2_path
        self.spectrum_keys = [
            'spectrum_integrated',
            'spectrum_real_packets',
            'spectrum_real_packets_reabsorbed',
            'spectrum_virtual_packets'
        ]
        self.data = {}

    def setup(self):
        for ref_name, file_path in [('Ref1', self.ref1_path), ('Ref2', self.ref2_path)]:
            self.data[ref_name] = {}
            try:
                with pd.HDFStore(file_path) as hdf:
                    for key in self.spectrum_keys:
                        full_key = f"simulation/spectrum_solver/{key}"
                        self.data[ref_name][key] = {
                            'wavelength': np.array(hdf[f'{full_key}/wavelength']),
                            'luminosity': np.array(hdf[f'{full_key}/luminosity'])
                        }
            except FileNotFoundError:
                print(f"Warning: File not found at {file_path}")
            except KeyError as e:
                print(f"Warning: Key {e} not found in {file_path}")

    def plot_matplotlib(self):
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.1, wspace=0.3)

        for idx, key in enumerate(self.spectrum_keys):
            row = (idx // 2) * 2
            col = idx % 2
            
            ax_luminosity = fig.add_subplot(gs[row, col])
            ax_residuals = fig.add_subplot(gs[row+1, col], sharex=ax_luminosity)
            
            # Plot luminosity
            for ref_name, linestyle in [('Ref1', '-'), ('Ref2', '--')]:
                if key in self.data[ref_name]:
                    wavelength = self.data[ref_name][key]['wavelength']
                    luminosity = self.data[ref_name][key]['luminosity']
                    ax_luminosity.plot(wavelength, luminosity, linestyle=linestyle, label=f'{ref_name} Luminosity')
            
            ax_luminosity.set_ylabel('Luminosity')
            ax_luminosity.set_title(f'Luminosity for {key}')
            ax_luminosity.legend()
            ax_luminosity.grid(True)
            
            # Plot fractional residuals
            if key in self.data['Ref1'] and key in self.data['Ref2']:
                wavelength = self.data['Ref1'][key]['wavelength']
                luminosity_ref1 = self.data['Ref1'][key]['luminosity']
                luminosity_ref2 = self.data['Ref2'][key]['luminosity']
                
                # Calculate fractional residuals
                with np.errstate(divide='ignore', invalid='ignore'):
                    fractional_residuals = np.where(luminosity_ref1 != 0, (luminosity_ref2 - luminosity_ref1) / luminosity_ref1, 0)
                
                ax_residuals.plot(wavelength, fractional_residuals, label='Fractional Residuals', color='purple')
                ax_residuals.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add a horizontal line at y=0
            
            ax_residuals.set_xlabel('Wavelength')
            ax_residuals.set_ylabel('Fractional Residuals')
            ax_residuals.legend()
            ax_residuals.grid(True)
            
            # Remove x-axis labels from upper plot
            ax_luminosity.tick_params(axis='x', labelbottom=False)
            
            # Only show x-label for bottom plots
            if row != 2:
                ax_residuals.tick_params(axis='x', labelbottom=False)
        
        plt.suptitle('Comparison of Spectrum Solvers with Fractional Residuals', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    def plot_plotly(self):
        # Create figure with shared x-axes
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=[
                'Luminosity for spectrum_integrated', 'Luminosity for spectrum_real_packets',
                'Fractional Residuals', 'Fractional Residuals',
                'Luminosity for spectrum_real_packets_reabsorbed', 'Luminosity for spectrum_virtual_packets',
                'Fractional Residuals', 'Fractional Residuals'
            ],
            vertical_spacing=0.07, 
            horizontal_spacing=0.08,  # Reduced from 0.15
            row_heights=[0.3, 0.15] * 2,
            shared_xaxes=True,

        )

        # Plot each spectrum type and its residuals
        for idx, key in enumerate(self.spectrum_keys):
            plot_col = idx % 2 + 1
            plot_row = (idx // 2) * 2 + 1
            
            # Store x-range for shared axis
            x_range = None
            
            # Plot luminosity traces
            for ref_name, line_style in [('Ref1', 'solid'), ('Ref2', 'dash')]:
                if key in self.data[ref_name]:
                    wavelength = self.data[ref_name][key]['wavelength']
                    luminosity = self.data[ref_name][key]['luminosity']
                    
                    if x_range is None:
                        x_range = [min(wavelength), max(wavelength)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=wavelength,
                            y=luminosity,
                            mode='lines',
                            name=f'{ref_name} - {key}',
                            line=dict(dash=line_style),
                        ),
                        row=plot_row,
                        col=plot_col
                    )
            
            # Plot residuals
            if key in self.data['Ref1'] and key in self.data['Ref2']:
                wavelength = self.data['Ref1'][key]['wavelength']
                luminosity_ref1 = self.data['Ref1'][key]['luminosity']
                luminosity_ref2 = self.data['Ref2'][key]['luminosity']
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    fractional_residuals = np.where(
                        luminosity_ref1 != 0,
                        (luminosity_ref2 - luminosity_ref1) / luminosity_ref1,
                        0
                    )
                
                fig.add_trace(
                    go.Scatter(
                        x=wavelength,
                        y=fractional_residuals,
                        mode='lines',
                        name=f'Residuals - {key}',
                        line=dict(color='purple'),
                    ),
                    row=plot_row + 1,
                    col=plot_col
                )
                
                fig.add_hline(
                    y=0,
                    line=dict(color='black', dash='dash', width=0.8),
                    row=plot_row + 1,
                    col=plot_col
                )

            # Update axes properties
            fig.update_xaxes(
                title_text="",
                showticklabels=False,
                row=plot_row,
                col=plot_col,
                gridcolor='lightgrey',
                showgrid=True,
                range=x_range
            )
            
            # Show x-axis for bottom plots
            fig.update_xaxes(
                title_text="Wavelength",
                row=plot_row + 1,
                col=plot_col,
                gridcolor='lightgrey',
                showgrid=True,
                range=x_range
            )
            
            fig.update_yaxes(
                title_text="Luminosity",
                row=plot_row,
                col=plot_col,
                gridcolor='lightgrey',
                showgrid=True
            )
            fig.update_yaxes(
                title_text="Fractional Residuals",
                row=plot_row + 1,
                col=plot_col,
                gridcolor='lightgrey',
                showgrid=True
            )

        # Update layout with minimal padding
        fig.update_layout(
            title='Comparison of Spectrum Solvers with Fractional Residuals',
            height=900,
            width=1200,
            showlegend=True,
            margin=dict(t=50, b=30, l=50, r=30),
            plot_bgcolor='rgba(240, 240, 255, 0.3)',
        )

        # Make subplot titles smaller and closer to plots
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=10)
            annotation['y'] = annotation['y'] - 0.02

        fig.show()

class ReferenceComparer:
    def __init__(self, ref1_hash=None, ref2_hash=None, print_path=False):
        assert not ((ref1_hash is None) and (ref2_hash is None)), "One hash can not be None"
        self.ref1_hash = ref1_hash
        self.ref2_hash = ref2_hash
        self.print_path = print_path
        self.test_table_dict = {}
        self.file_manager = FileManager()
        self.file_setup = None
        self.diff_analyzer = None
        self.hdf_comparator = None

    def setup(self):
        self.file_manager.setup()
        self.file_setup = FileSetup(self.file_manager, self.ref1_hash, self.ref2_hash)
        self.diff_analyzer = DiffAnalyzer(self.file_manager)
        self.hdf_comparator = HDFComparator(print_path=self.print_path)
        self.file_setup.setup()
        self.ref1_path = self.file_manager.get_temp_path("ref1")
        self.ref2_path = self.file_manager.get_temp_path("ref2")
        self.dcmp = dircmp(self.ref1_path, self.ref2_path)

    def teardown(self):
        self.file_manager.teardown()

    def compare(self, print_diff=False):
        if print_diff:
            self.diff_analyzer.print_diff_files(self.dcmp)
        self.compare_hdf_files()
        
        # Update test_table_dict with added and deleted keys
        for name, results in self.test_table_dict.items():
            ref1_keys = set(results.get("ref1_keys", []))
            ref2_keys = set(results.get("ref2_keys", []))
            results["added_keys"] = list(ref2_keys - ref1_keys)
            results["deleted_keys"] = list(ref1_keys - ref2_keys)

    def compare_hdf_files(self):
        for root, _, files in os.walk(self.ref1_path):
            for file in files:
                file_path = Path(file)
                if file_path.suffix in ('.h5', '.hdf5'):
                    rel_path = Path(root).relative_to(self.ref1_path)
                    ref2_file_path = self.ref2_path / rel_path / file
                    if ref2_file_path.exists():
                        self.summarise_changes_hdf(file, root, ref2_file_path.parent)

    def summarise_changes_hdf(self, name, path1, path2):
        self.test_table_dict[name] = {
            "path": get_relative_path(path1, self.file_manager.temp_dir / "ref1")
        }
        results = self.hdf_comparator.summarise_changes_hdf(name, path1, path2)
        self.test_table_dict[name].update(results)
        
        # Store keys for both references
        self.test_table_dict[name]["ref1_keys"] = results.get("ref1_keys", [])
        self.test_table_dict[name]["ref2_keys"] = results.get("ref2_keys", [])

    def display_hdf_comparison_results(self):
        for name, results in self.test_table_dict.items():
            print(f"Results for {name}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            print()

    def get_temp_dir(self):
        return self.file_manager.temp_dir

    def generate_graph(self, option):
        print("Generating graph with updated hovertemplate")
        if option not in ["different keys same name", "different keys"]:
            raise ValueError("Invalid option. Choose 'different keys same name' or 'different keys'.")

        data = []
        for name, results in self.test_table_dict.items():
            if option == "different keys same name":
                value = results.get("identical_keys_diff_data", 0)
                if value > 0:
                    diff_data = results["identical_name_different_data_dfs"]
                    keys = list(diff_data.keys())
                    # Calculate max relative difference for each key
                    rel_diffs = [diff_data[key].abs().max().max() for key in keys]
                    data.append((name, value, keys, rel_diffs))
            else:  # "different keys"
                value = results.get("different_keys", 0)
                if value > 0:
                    added = list(results.get("added_keys", []))
                    deleted = list(results.get("deleted_keys", []))
                    data.append((name, value, added, deleted))

        if not data:
            return None

        fig = go.Figure()

        # Extract filenames from the full paths
        filenames = [item[0].split('/')[-1] for item in data]

        for item in data:
            name = item[0]
            if option == "different keys same name":
                _, value, keys, rel_diffs = item
                if rel_diffs:
                    max_diff = max(rel_diffs)
                    normalized_diffs = [diff / max_diff for diff in rel_diffs]
                    colors = [pc.sample_colorscale('Blues', diff)[0] for diff in normalized_diffs]
                else:
                    colors = ['rgb(220, 220, 255)'] * len(keys)
                    rel_diffs = [0] * len(keys)  # Set all differences to 0

                fig.add_trace(go.Bar(
                    y=[name] * len(keys),
                    x=[1] * len(keys),
                    orientation='h',
                    name=name,
                    text=keys,
                    customdata=rel_diffs,
                    marker_color=colors,
                    hoverinfo='text',
                    hovertext=[f"{name}<br>Key: {key}<br>Max relative difference: {diff:.2e}<br>(Versions differ by {diff:.1%})" 
                               for key, diff in zip(keys, rel_diffs)]
                ))
            else:  # "different keys"
                _, _, added, deleted = item
                colors_added = [f'rgb(0, {random.randint(100, 255)}, 0)' for _ in added]
                colors_deleted = [f'rgb({random.randint(100, 255)}, 0, 0)' for _ in deleted]
                fig.add_trace(go.Bar(
                    y=[name] * len(added),
                    x=[1] * len(added),
                    orientation='h',
                    name=f"{name} (Added)",
                    text=added,
                    hovertemplate='%{y}<br>Added Key: %{text}<extra></extra>',
                    marker_color=colors_added
                ))
                fig.add_trace(go.Bar(
                    y=[name] * len(deleted),
                    x=[1] * len(deleted),
                    orientation='h',
                    name=f"{name} (Deleted)",
                    text=deleted,
                    hovertemplate='%{y}<br>Deleted Key: %{text}<extra></extra>',
                    marker_color=colors_deleted
                ))

        fig.update_layout(
            title=f"{'Different Keys with Same Name' if option == 'different keys same name' else 'Different Keys'} Comparison",
            barmode='stack',
            height=max(300, len(data) * 40),  # Adjust height based on number of files
            xaxis_title="Number of Keys",
            yaxis=dict(
                title='',
                tickmode='array',
                tickvals=list(range(len(filenames))),
                ticktext=filenames,
                showgrid=False
            ),
            showlegend=False,
            bargap=0.1,
            bargroupgap=0.05,
            margin=dict(l=200)  # Increase left margin to accommodate longer filenames
        )

        # Remove the text on the right side of the bars
        fig.update_traces(textposition='none')

        # Add a color bar to show the scale
        if any(item[3] for item in data if option == "different keys same name"):
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="Relative Difference",
                    tickvals=[0, 0.5, 1],
                    ticktext=["Low", "Medium", "High"],
                    lenmode="fraction",
                    len=0.75,
                )
            )

        return fig
    
    @classmethod
    def compare_testspectrumsolver_hdf(cls, custom_ref1_path=None, custom_ref2_path=None):
        ref1_path = custom_ref1_path or "tardis/spectrum/tests/test_spectrum_solver/test_spectrum_solver/TestSpectrumSolver.h5"
        ref2_path = custom_ref2_path or "tardis/spectrum/tests/test_spectrum_solver/test_spectrum_solver/TestSpectrumSolver.h5"
        
        comparator = SpectrumSolverComparator(ref1_path, ref2_path)
        comparator.setup()
        
        comparator.plot_matplotlib()
        
        comparator.plot_plotly()

