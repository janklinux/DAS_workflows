import io
import os
import gzip
import subprocess
from pymongo import MongoClient
from ase.io import read as aseread
from ase.io import write as asewrite
from das_workflows.helpers import env_chk
from monty.serialization import dumpfn
from fireworks import Firework, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from pymatgen.core.structure import Structure
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from custodian import Custodian
from custodian.custodian import Job
from custodian.vasp.handlers import VaspErrorHandler, MeshSymmetryErrorHandler, UnconvergedErrorHandler, \
    NonConvergingErrorHandler, PotimErrorHandler, PositiveEnergyErrorHandler, FrozenJobErrorHandler, StdErrHandler
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator


@explicit_serialize
class FewstepsFW(Firework):
    def __init__(self, structure, vasp_input_set, vasp_cmd, name):
        """
        Standard static calculation Firework for a structure.

        Args:
            structure (Structure): Input structure
            vasp_input_set (VaspInputSet): input set to use
            vasp_cmd (str): Command to run vasp.
            name: the name of the workflow
        """

        t = list()
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(RunVaspFewCustodian(vasp_cmd=vasp_cmd))
        super(FewstepsFW, self).__init__(t, name=name)


@explicit_serialize
class RunVaspFewCustodian(FiretaskBase):
    """
    Run VASP using custodian "on rails", fixes most runtime errors, uses not all handlers
    but default validators from custodian package

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP
    """

    required_params = ['vasp_cmd']
    optional_params = []

    def run_task(self, fw_spec):
        vasp_cmd = env_chk(self['vasp_cmd'], fw_spec)
        handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(),
                    PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()]
        validators = [VasprunXMLValidator(), VaspFilesValidator()]

        c = Custodian(handlers, [VaspJob(vasp_cmd=vasp_cmd)], validators=validators, max_errors=5)
        c.run()


@explicit_serialize
class SinglePointFW(Firework):
    def __init__(self, structure, vasp_input_set, vasp_cmd, name):
        """
        Standard static calculation Firework for a structure.

        Args:
            structure (Structure): Input structure
            vasp_input_set (VaspInputSet): input set to use
            vasp_cmd (str): Command to run vasp.
            name: the name of the workflow
        """

        t = list()
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd))
        super(SinglePointFW, self).__init__(t, name=name)


@explicit_serialize
class RunVaspCustodian(FiretaskBase):
    """
    Run VASP using custodian "on rails", fixes most runtime errors, uses default handlers
    and validators from custodian package

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP
    """

    required_params = ['vasp_cmd']
    optional_params = []

    def run_task(self, fw_spec):
        vasp_cmd = env_chk(self['vasp_cmd'], fw_spec)
        handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
                    NonConvergingErrorHandler(), PotimErrorHandler(),
                    PositiveEnergyErrorHandler(), FrozenJobErrorHandler(), StdErrHandler()]
        validators = [VasprunXMLValidator(), VaspFilesValidator()]

        c = Custodian(handlers, [VaspJob(vasp_cmd=vasp_cmd)], validators=validators, max_errors=5)
        c.run()


@explicit_serialize
class CoordinationTracker(Firework):
    def __init__(self, input_trajectory, track_species, keep_trajectory, max_coord_number):
        """
        Standard static calculation Firework for a structure.

        Args:
            input_trajectory: filename of the trajectory
            track_species(list): species to track
        """

        t = list()
        t.append(RunCoordinationTracker(trajectory_filename=input_trajectory, track_species=track_species,
                                        keep_trajectory=keep_trajectory, max_coord_number=max_coord_number))
        super(CoordinationTracker, self).__init__(t, name='CoordinationTracker')


@explicit_serialize
class RunCoordinationTracker(FiretaskBase):
    """
    Run VASP using custodian "on rails", fixes most runtime errors, uses default handlers
    and validators from custodian package

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP
    """

    required_params = ['trajectory_filename', 'track_species', 'keep_trajectory', 'max_coord_number']
    optional_params = []

    def run_task(self, fw_spec):
        previous_dir = '/'.join(self['trajectory_filename'].split('/')[:-1])
        suffix = self['trajectory_filename'].split('/')[-1].split('.')[0].split('_')[-1]

        dirty_atoms = aseread(self['trajectory_filename'], index=':')

        new_data = dict()

        track_species = self['track_species']

        lgf = LocalGeometryFinder()
        lgf.setup_parameters(structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE)

        for idx, (atoms) in enumerate(dirty_atoms):
            track_index = []
            for at in atoms:
                if at.symbol in track_species:
                    track_index.append(at.index)

            for new_idx in track_index:
                if new_idx not in new_data:
                    new_data[new_idx] = dict()

            lgf.setup_structure(AseAtomsAdaptor().get_structure(atoms))

            try:
                se = lgf.compute_structure_environments(only_indices=track_index, max_cn=self['max_coord_number'])
            except RuntimeError:
                print('seife gelutscht')
                return

            for counter, site_envs in enumerate(se.ce_list):
                if site_envs is not None:
                    for crd_num in range(2, self['max_coord_number'] + 1):
                        if crd_num in site_envs:
                            for ce in site_envs[crd_num]:
                                for symbol, info in ce.minimum_geometries():
                                    if symbol not in new_data[counter]:
                                        new_data[counter][symbol] = list()
                                    if int(info['detailed_voronoi_index']['index']) == 0:
                                        csm = float(info["other_symmetry_measures"]["csm_wocs_ctwcc"])
                                        new_data[counter][symbol].append(float(csm))

            # adjust all envs to match longest entry and pad up missing envs
            all_envs = list()
            max_entries = 0
            for my_idx in new_data:
                if len(new_data[my_idx]) > max_entries:
                    max_entries = len(new_data[my_idx])
                    for symbol in new_data[my_idx]:
                        if symbol not in all_envs:
                            all_envs.append(symbol)

            # print('all env:', all_envs)

            for my_idx in new_data:
                if len(new_data[my_idx]) != max_entries:
                    for symbol in all_envs:
                        if symbol not in new_data[my_idx]:
                            new_data[my_idx][symbol] = list()

                for symbol in all_envs:
                    if len(new_data[my_idx][symbol]) != idx + 1:
                        new_data[my_idx][symbol].append(float(110))

        os.chdir(previous_dir)
        dumpfn(new_data, 'coordination_{}.json'.format(suffix))
        os.system('gzip coordination_{}.json'.format(suffix))
        if not self['keep_trajectory']:
            os.unlink(self['trajectory_filename'])


@explicit_serialize
class SinglePointFWNoPosEnErr(Firework):
    def __init__(self, structure, vasp_input_set, vasp_cmd, name):
        """
        Standard static calculation Firework for a structure.

        Args:
            structure (Structure): Input structure
            vasp_input_set (VaspInputSet): input set to use
            vasp_cmd (str): Command to run vasp.
            name: the name of the workflow
        """

        t = list()
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(RunVaspCustodianNoPosEnErr(vasp_cmd=vasp_cmd))
        super(SinglePointFWNoPosEnErr, self).__init__(t, name=name)


@explicit_serialize
class RunVaspCustodianNoPosEnErr(FiretaskBase):
    """
    Run VASP using custodian "on rails", fixes most runtime errors, uses default handlers
    and validators from custodian package

    Required params:
        vasp_cmd (str): the name of the full executable for running VASP
    """

    required_params = ['vasp_cmd']
    optional_params = []

    def run_task(self, fw_spec):
        vasp_cmd = env_chk(self['vasp_cmd'], fw_spec)
        handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
                    NonConvergingErrorHandler(), PotimErrorHandler(),
                    FrozenJobErrorHandler(), StdErrHandler()]
        validators = [VasprunXMLValidator(), VaspFilesValidator()]

        c = Custodian(handlers, [VaspJob(vasp_cmd=vasp_cmd)], validators=validators, max_errors=5)
        c.run()


class VaspJob(Job):
    """
    A basic VASP job
    """

    def __init__(self, vasp_cmd):
        """
        Get/set variables for a simple VASP job
        Args:
            vasp_cmd (str): Command to run vasp
        """
        self.run_dir = os.getcwd()
        self.vasp_cmd = vasp_cmd
        self.std_out = 'vasp.out'  # compatible to handlers
        self.std_err = 'std_err.txt'  # compatible to handlers

    def setup(self):
        """
        We don't have to do anything here
        """
        pass

    def run(self):
        """
        Runs VASP

        Returns:
            open subprocess for monitoring by custodian
        """

        with open(self.std_out, 'w') as sout, open(self.std_err, 'w', buffering=1) as serr:
            p = subprocess.Popen(self.vasp_cmd.split(), stdout=sout, stderr=serr)
        return p

    def postprocess(self):
        """
        For now, gzip all files in the directory we ran VASP in
        """

        for file in os.listdir(self.run_dir):
            with open(file, 'rb') as fin:
                with gzip.open(file + '.gz', 'wb') as fout:
                    fout.write(fin.read())
            os.unlink(file)


@explicit_serialize
class AddToDbTask(FiretaskBase):
    """
    Task insert results into a database if energy and forces exceed specification

    Required:
        db_file (str): absolute path to file containing the database credentials
        force_thresh (float): Threshold for any force component above which the result is added to the training db
    """

    required_params = ['force_thresh', 'energy_thresh', 'db_info', 'lammps_energy']
    optional_params = []

    def run_task(self, fw_spec):
        """
        Does the work by connecting to db, parsing the results, checking the thresholds and adding the
        structure to the db if needed

        Args:
            fw_spec: fireworks specifics

        Returns:
            nothing

        """

        connection = None

        if 'ssl' in self['db_info']:
            if self['db_info']['ssl'].lower() == 'true':
                try:
                    connection = MongoClient(host=self['db_info']['host'], port=self['db_info']['port'],
                                             username=self['db_info']['user'], password=self['db_info']['password'],
                                             ssl=True, tlsCAFile=self['db_info']['ssl_ca_certs'],
                                             ssl_certfile=self['db_info']['ssl_certfile'])
                except ConnectionError:
                    raise ConnectionError('Mongodb connection failed')
            else:
                try:
                    connection = MongoClient(host=self['db_info']['host'], port=self['db_info']['port'],
                                             username=self['db_info']['user'], password=self['db_info']['password'],
                                             ssl=False)
                except ConnectionError:
                    raise ConnectionError('Mongodb connection failed')

        if connection is None:
            raise ConnectionAbortedError('Connection failure, check internal routines')

        db = connection[self['db_info']['database']]
        try:
            db.authenticate(self['db_info']['user'], self['db_info']['password'])
        except ConnectionRefusedError:
            raise ConnectionRefusedError('Mongodb authentication failed')
        # collection = db[self['db_info']['structure_collection']]

        # get the directory we parse files in
        run_dir = os.getcwd()

        vrun = os.path.join(run_dir, 'vasprun.xml.gz')
        # orun = os.path.join(run_dir, 'OUTCAR.gz')

        run = Vasprun(vrun)
        if not run.converged:
            return
        # runo = Outcar(orun)
        atoms = aseread(vrun)
        xyz = ''
        file = io.StringIO()
        asewrite(filename=file, images=atoms, format='xyz')
        file.seek(0)
        for f in file:
            xyz += f
        file.close()
