import os
import gzip
import shutil
import tempfile
import subprocess
import numpy as np
from ase.io import read as ase_read
from ase.io import write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor
from custodian.custodian import Job


class TGapPhaseDiagramJob(Job):
    """
    Class to run turbogap for phase diagrams
    """

    def __init__(self, params, fw_spec):
        """
        Init class, sets parameters and does checks

        Args:
            params(dict): all turbogap parameters
            fw_spec(dict): fireworks specs
        """

        self.params = params
        self.fw_spec = fw_spec
        self.xyz = params['xyz']
        self.ratio = params['ratio']
        self.run_dir = params['run_dir']
        self.base_dir = params['base_dir']
        self.initial_run = params['initial_run']
        self.gap_files_dir = params['gap_files_dir']
        self.keep_trajectories = params['keep_trajectories']

    def setup(self):
        """
        Prepare for turbogap run, writes and links input files

        Returns:
            nothing

        """
        os.chdir(self.base_dir)

        if self.ratio is not None:
            if not os.path.isdir(self.ratio):
                os.mkdir(self.ratio)
            os.chdir(self.ratio)
        else:
            self.ratio = ''

        if not os.path.isdir(self.run_dir):
            os.mkdir(self.run_dir)
        os.chdir(self.run_dir)

        with open('input', 'w') as f:
            for line in self.params['input']:
                f.write(line + '\n')

        if self.initial_run:
            if not os.path.islink('gap_files'):
                os.symlink(os.path.join(self.gap_files_dir, 'gap_files'), 'gap_files')
            with open('initial.xyz', 'w') as f:
                for line in self.xyz:
                    f.write(line)

    def run(self):
        """
        Runs the job

        Returns:
            open subprocess
        """
        os.chdir(os.path.join(self.base_dir, self.ratio, self.run_dir))

        cmd = '{}'.format(self.params['run_cmd'])

        try:
            with open('std_err', 'w') as serr, open('std_out', 'w') as sout:
                p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
        except FileNotFoundError:
            raise FileNotFoundError('Command execution failed, check std_err.')
        return p

    def postprocess(self):
        """
        Possible tasks after run has finished, does nothing for now

        Returns:
            nothing

        """
        os.chdir(os.path.join(self.base_dir, self.ratio, self.run_dir))

        last_atoms = ase_read('trajectory_out.xyz', index=':')[-1]
        ase_write(filename='initial.xyz', images=last_atoms, format='extxyz')

        with open('std_out', 'rb') as f_in:
            with gzip.open('stdout_{}.gz'.format(self.params['store_name']), 'w') as f_out:
                f_out.write(f_in.read())
        with open('thermo.log', 'rb') as f_in:
            with gzip.open('thermo_{}.gz'.format(self.params['store_name']), 'w') as f_out:
                f_out.write(f_in.read())
        if self.keep_trajectories:
            with open('trajectory_out.xyz', 'rb') as f_in:
                with gzip.open('trajectory_{}.gz'.format(self.params['store_name']), 'w') as f_out:
                    f_out.write(f_in.read())

        os.unlink('std_out')
        os.unlink('thermo.log')
        os.unlink('trajectory_out.xyz')


class TGapSequentialJob(Job):
    """
    Class to run turbogap in sequential mode, requires setting parents properly
    """

    def __init__(self, params, fw_spec):
        """
        Init class, sets parameters and does checks

        Args:
            params(dict): all turbogap parameters
            fw_spec(dict): fireworks specs
        """

        self.params = params
        self.fw_spec = fw_spec
        self.xyz = params['xyz']
        self.use_tmp = params['use_tmp']
        self.run_dir = params['run_dir']
        self.base_dir = params['base_dir']
        self.initial_run = params['initial_run']
        self.gap_files_dir = params['gap_files_dir']
        self.keep_trajectories = params['keep_trajectories']
        self.add_vasp_static_wf = params['add_vasp_static_wf']
        self.energy_window = params['energy_window']
        self.window_height = params['window_height']
        self.gradient_check = params['gradient_check']
        self.gradient_value = params['gradient_value']
        self.coordination_tracker = params['coordination_tracker']
        self.track_species = params['track_species']
        self.incar_mod = params['incar_mod']

    def setup(self):
        """
        Prepare for turbogap run, writes and links input files

        Returns:
            nothing

        """
        pass

    def run(self):
        """
        Runs the job

        Returns:
            open subprocess
        """
        if self.use_tmp:
            os.chdir('/tmp')
            tempdir = tempfile.mktemp()
            os.mkdir(tempdir)
            os.chdir(tempdir)

            if os.path.isdir(os.path.join(self.base_dir, self.run_dir)):
                os.mkdir(self.run_dir)
                for file in os.listdir(str(os.path.join(self.base_dir, self.run_dir))):
                    shutil.copy(str(os.path.join(self.base_dir, self.run_dir, file)), self.run_dir)
            else:
                os.mkdir(self.run_dir)
        else:
            os.chdir(self.base_dir)
            if not os.path.isdir(os.path.join(self.base_dir, self.run_dir)):
                os.mkdir(self.run_dir)

        os.chdir(self.run_dir)

        with open('input', 'w') as f:
            for line in self.params['input']:
                f.write(line + '\n')

        if not os.path.islink('gap_files'):
            os.symlink(os.path.join(self.gap_files_dir, 'gap_files'), 'gap_files')

        if self.initial_run:
            with open('initial.xyz', 'w') as f:
                for line in self.xyz:
                    f.write(line)

        cmd = '{}'.format(self.params['run_cmd'])

        try:
            with open('std_err', 'w') as serr, open('std_out', 'w') as sout:
                p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
        except FileNotFoundError:
            raise FileNotFoundError('Command execution failed, check std_err.')

        p.communicate()

        dirty_atoms = ase_read('trajectory_out.xyz', index=':')
        last_atoms = dirty_atoms[-1]
        if np.any([np.any(k) for k in [np.isnan(x) for x in last_atoms.get_positions()]]):
            shutil.copy('initial.xyz', 'initial_preNaN.xyz')
            os.unlink('initial.xyz')
            raise ValueError('NaN Seife')

        if self.params['store_name'] == 'initial-relax':  # bussi workaround
            # print('BUSSI')
            for key in ['velocities', 'forces', 'local_energy', 'fix_atoms']:
                if key in last_atoms.arrays:
                    del last_atoms.arrays[key]
        ase_write(filename='initial.xyz', images=last_atoms, format='extxyz')

        # with open('std_out', 'rb') as f_in:
        #     with gzip.open('stdout_{}.gz'.format(self.params['store_name']), 'w') as f_out:
        #         f_out.write(f_in.read())
        with open('thermo.log', 'rb') as f_in:
            with gzip.open('thermo_{}.gz'.format(self.params['store_name']), 'w') as f_out:
                f_out.write(f_in.read())
        with open('trajectory_out.xyz', 'rb') as f_in:
            with gzip.open('trajectory_{}.xyz.gz'.format(self.params['store_name']), 'w') as f_out:
                f_out.write(f_in.read())

        if np.any([np.any(k) for k in [np.isnan(x) for x in dirty_atoms[-1].get_positions()]]):
            raise ValueError('NaN found...')

        result = list()

        if self.add_vasp_static_wf:
            my_atoms = dirty_atoms[-1].copy()
            for key in ['fix_atoms', 'forces', 'local_energy', 'velocities']:
                if key in my_atoms.arrays:
                    del my_atoms.arrays[key]
            my_structure = AseAtomsAdaptor().get_structure(my_atoms).as_dict()
            os.unlink('trajectory_out.xyz')
            if 'potcar_mod' in self.params:
                my_pots = self.params['potcar_mod']
            else:
                my_pots = None

            result.append(dict({'struct': my_structure, 'incar_mod': self.incar_mod, 'defuse_children': False,
                                'potcar_mod': my_pots, 'kpt_dict': self.params['kpt_dict'],
                                'symptom': 'add_static',
                                'vasp_cmd': self.params['vasp_cmd'],
                                'wf_name': self.params['store_name'] + '_' + self.params['run_dir'],
                                'dft_pad': self.params['dft_pad']}))

        if self.coordination_tracker:
            has_species = False
            for spec in self.track_species:
                if spec in dirty_atoms[0].get_chemical_symbols():
                    has_species = True

            if not has_species:
                print('SPECIES NOT PRESENT for tracker')
                return None
            else:
                result.append(dict({'input_trajectory': '{}/trajectory_{}.xyz.gz'.format(
                    os.path.join(str(os.path.join(self.base_dir, self.run_dir))), self.params['store_name']),
                                    'track_species': self.track_species,
                                    'keep_trajectory': self.keep_trajectories,
                                    'struct': None, 'incar_mod': self.incar_mod, 'defuse_children': False,
                                    'symptom': 'crd_track',
                                    'max_coord_number': self.params['max_coord_number'],
                                    'wf_name': self.params['store_name'] + '_' + self.params['run_dir'],
                                    'crd_pad': self.params['crd_pad']}))

        if self.gradient_check:
            add_index = None
            gradient = np.array([np.gradient([e for e in [atoms.get_potential_energy() for atoms in dirty_atoms]])])
            if np.any(np.array([gradient > self.gradient_value])):
                where = np.where(np.array(gradient) > self.gradient_value)[1]
                if not len(where) > 3:
                    add_index = where[:]
                else:
                    add_index = where[0:3]

            if add_index is not None:
                with open('trajectory_out.xyz', 'rb') as f_in:
                    with gzip.open('trajectory_grad_{}.xyz.gz'.format(self.params['store_name']),
                                   'w') as f_out:
                        f_out.write(f_in.read())

                if self.params['add_gradient_dft']:
                    defuse_children = True
                    for index in add_index:
                        if 'potcar_mod' in self.params:
                            my_pots = self.params['potcar_mod']
                        else:
                            my_pots = None

                        result.append(dict({'struct': AseAtomsAdaptor().get_structure(dirty_atoms[index]).as_dict(),
                                            'incar_mod': self.incar_mod,
                                            'potcar_mod': my_pots,
                                            'kpt_dict': self.params['kpt_dict'],
                                            'symptom': 'gradient', 'add_gradient_dft': self.params['add_gradient_dft'],
                                            'defuse_children': defuse_children,
                                            'vasp_cmd': self.params['vasp_cmd'],
                                            'wf_name': self.params['store_name'] + '_' +
                                            self.params['run_dir'] + '_atoms_{}'.format(index),
                                            'dft_pad': self.params['dft_pad']}))

            # return None  # do not return here, it breaks the chain

        if self.energy_window:
            max_energy = np.amax([e for e in [atoms.get_potential_energy() / len(atoms) for atoms in dirty_atoms]])
            min_energy = np.amin([e for e in [atoms.get_potential_energy() / len(atoms) for atoms in dirty_atoms]])
            # initial_energy = dirty_atoms[0].get_potential_energy() / len(dirty_atoms[0])
            final_energy = dirty_atoms[-1].get_potential_energy() / len(dirty_atoms[-1])

            if np.abs(np.abs(max_energy) - np.abs(min_energy)) > np.abs(final_energy * self.window_height):
                with open('trajectory_out.xyz', 'rb') as f_in:
                    with gzip.open('trajectory_window_{}.xyz.gz'.format(self.params['store_name']),
                                   'w') as f_out:
                        f_out.write(f_in.read())

                if self.params['add_energy_dft']:
                    add_atoms = []
                    for en, at in reversed(sorted(
                            zip([e for e in [atoms.get_potential_energy() / len(atoms)
                                             for atoms in dirty_atoms]], dirty_atoms), key=lambda k: k[0])):
                        if en <= 0:
                            print(at.get_potential_energy(), at)
                            add_atoms.append(at)
                            if len(add_atoms) == 3:
                                break

                    defuse_children = True

                    for index, atoms in enumerate(add_atoms):
                        result.append(dict({'struct': AseAtomsAdaptor().get_structure(atoms).as_dict(),
                                            'incar_mod': self.incar_mod,
                                            'symptom': 'window', 'add_energy_dft': self.params['add_energy_dft'],
                                            'defuse_children': defuse_children,
                                            'vasp_cmd': self.params['vasp_cmd'],
                                            'wf_name': self.params['store_name'] + '_' +
                                            self.params['run_dir'] + '_step_{}'.format(index),
                                            'dft_pad': self.params['dft_pad']}))

        os.unlink('std_out')
        os.unlink('thermo.log')

        if not self.coordination_tracker and not self.keep_trajectories:
            os.unlink('trajectory_{}.xyz.gz'.format(self.params['store_name']))

        if os.path.isfile('trajectory_out.xyz'):
            os.unlink('trajectory_out.xyz')

        if self.use_tmp:
            if os.path.isdir(os.path.join(str(os.path.join(self.base_dir, self.run_dir)))):
                for file in os.listdir():
                    if os.path.isfile(file):
                        shutil.copy(file, str(os.path.join(self.base_dir, self.run_dir)))
            else:
                os.mkdir(os.path.join(str(os.path.join(self.base_dir, self.run_dir))))
                for file in os.listdir():
                    if os.path.isfile(file):
                        shutil.copy(file, str(os.path.join(self.base_dir, self.run_dir)))

        return result

    def postprocess(self):
        """
        Possible tasks after run has finished

        Returns:
            nothing

        """
        pass


class TGapConvergeJob(Job):
    """
    Class to run turbogap in sequential mode, requires setting parents properly
    """

    def __init__(self, params, fw_spec):
        """
        Init class, sets parameters and does checks

        Args:
            params(dict): all turbogap parameters
            fw_spec(dict): fireworks specs
        """

        self.fw_spec = fw_spec
        self.params = params
        self.xyz = params['xyz']
        self.use_tmp = params['use_tmp']
        self.run_dir = params['run_dir']
        self.base_dir = params['base_dir']
        self.initial_run = params['initial_run']
        self.gap_files_dir = params['gap_files_dir']
        self.keep_trajectories = params['keep_trajectories']
        self.add_vasp_static_wf = params['add_vasp_static_wf']
        self.energy_window = params['energy_window']
        self.window_height = params['window_height']
        self.gradient_check = params['gradient_check']
        self.gradient_value = params['gradient_value']
        self.coordination_tracker = params['coordination_tracker']
        self.track_species = params['track_species']
        self.incar_mod = params['incar_mod']
        self.relax_stepper = params['relax_stepper']

    def setup(self):
        """
        Prepare for turbogap run, writes and links input files

        Returns:
            nothing

        """
        pass

    def run(self):
        """
        Runs the job

        Returns:
            open subprocess
        """
        if self.use_tmp:
            os.chdir('/tmp')
            tempdir = tempfile.mktemp()
            os.mkdir(tempdir)
            os.chdir(tempdir)

            if os.path.isdir(os.path.join(self.base_dir, self.run_dir)):
                os.mkdir(self.run_dir)
                for file in os.listdir(str(os.path.join(self.base_dir, self.run_dir))):
                    shutil.copy(str(os.path.join(self.base_dir, self.run_dir, file)), self.run_dir)
            else:
                os.mkdir(self.run_dir)
        else:
            os.chdir(self.base_dir)
            if not os.path.isdir(os.path.join(self.base_dir, self.run_dir)):
                os.mkdir(self.run_dir)

        os.chdir(self.run_dir)

        atoms = None
        first_cycle = True
        dft_structures = list()
        for cycle in range(self.relax_stepper):
            moldyn_steps = None
            with open('input', 'w') as f_out:
                for line in self.params['input']:
                    if 'md_nsteps' in line:
                        moldyn_steps = int(line.split('=')[1])
                    f_out.write(line + '\n')

            if not os.path.islink('gap_files'):
                os.symlink(os.path.join(self.gap_files_dir, 'gap_files'), 'gap_files')

            if first_cycle:
                first_cycle = False
                with open('initial.xyz', 'w') as f:
                    for line in self.xyz:
                        f.write(line)
            else:
                ase_write(filename='initial.xyz', images=atoms)

            cmd = '{}'.format(self.params['run_cmd'])

            try:
                with open('std_err', 'w') as serr, open('std_out', 'w') as sout:
                    p = subprocess.Popen(cmd.split(), stdout=sout, stderr=serr)
            except FileNotFoundError:
                raise FileNotFoundError('Command execution failed, check std_err.')

            p.communicate()

            dirty_atoms = ase_read('trajectory_out.xyz', index=':')
            last_atoms = dirty_atoms[-1]
            if np.any([np.any(k) for k in [np.isnan(x) for x in last_atoms.get_positions()]]):
                shutil.copy('initial.xyz', 'initial_preNaN.xyz')
                os.unlink('initial.xyz')
                raise ValueError('NaN Seife')

            dft_structures.append(AseAtomsAdaptor().get_structure(last_atoms))

            if len(dirty_atoms) < moldyn_steps:
                break

            atoms = last_atoms

        result = list()
        for index, structure in enumerate(dft_structures):
            if 'potcar_mod' in self.params:
                my_pots = self.params['potcar_mod']
            else:
                my_pots = None

            result.append(dict({'struct': structure.as_dict(), 'incar_mod': self.incar_mod, 'symptom': 'converge_flow',
                                'defuse_children': False, 'potcar_mod': my_pots, 'kpt_dict': self.params['kpt_dict'],
                                'vasp_cmd': self.params['vasp_cmd'],
                                'wf_name': self.params['store_name'] + '_' + self.params['run_dir']
                                           + '_step_{}'.format(index),
                                'dft_pad': self.params['dft_pad']}))

        return result

    def postprocess(self):
        """
        Possible tasks after run has finished

        Returns:
            nothing

        """
        pass
