import abc
from fireworks import Workflow, LaunchPad
from fireworks.core.firework import FWAction, Firework, FiretaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from custodian import Custodian
from das_workflows.turbogap.turbogap import TGapPhaseDiagramJob, TGapSequentialJob, TGapConvergeJob
from das_workflows.firetasks.vasp import SinglePointFW, CoordinationTracker
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.core.structure import Structure


class TrainBase(FiretaskBase):
    """
    Base class to train potentials, we inherit from it
    """
    @abc.abstractmethod
    def get_job(self):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job()
        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        c.run()


@explicit_serialize
class PotentialTraining(TrainBase):
    """
    Class to train a potential
    """
    required_params = ['train_params', 'for_validation', 'db_info']
    optional_params = ['al_info']

    def get_job(self):
        return TrainJob(train_params=self['train_params'], for_validation=self['for_validation'],
                        db_info=self['db_info'])

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


class LammpsBase(FiretaskBase):
    """
    Base class to run LAMMPS, for inheritance
    """
    @abc.abstractmethod
    def get_job(self, fw_spec):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job(fw_spec)
        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        c.run()
        # return FWAction(additions=job.get_vasp_static_dft(job.get_lammps_energy()))


@explicit_serialize
class Lammps(LammpsBase):
    """
    Class to run LAMMPS
    """
    required_params = ['lammps_params']
    optional_params = []

    def get_job(self, fw_spec):
        return LammpsJob(lammps_params=self['lammps_params'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


@explicit_serialize
class LammpsPhaseDiagram(LammpsBase):
    """
    Class to run LAMMPS
    """
    required_params = ['lammps_params']
    optional_params = []

    def get_job(self, fw_spec):
        return LammpsPhaseDiagramJob(lammps_params=self['lammps_params'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


class TGapBase(FiretaskBase):
    """
    Base class to run turbogap, for inheritance
    """
    @abc.abstractmethod
    def get_job(self, fw_spec):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job(fw_spec)
        # c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        data_chunk = job.run()
        if data_chunk is not None:
            add_wf = list()
            dft_pad = None
            add_stuff = False
            defuse_children = False
            for data in data_chunk:
                if data['symptom'] == 'has_nan':
                    raise ValueError('not supposed to be here')

                elif data['symptom'] == 'crd_track':
                    add_pad = LaunchPad().from_dict(data['crd_pad'])
                    my_wf = Workflow([CoordinationTracker(input_trajectory=data['input_trajectory'],
                                                          track_species=data['track_species'],
                                                          keep_trajectory=data['keep_trajectory'],
                                                          max_coord_number=data['max_coord_number'])],
                                     name='tracker {}'.format(data['wf_name']),
                                     metadata={'added': 'tracker {}'.format(data['wf_name'])})
                    add_pad.add_wf(my_wf)

                elif data['symptom'] == 'add_static':
                    structure = Structure.from_dict(data['struct'])
                    incar_set = MPStaticSet(structure)
                    in_dict = incar_set.as_dict()
                    in_dict.update({'user_kpoints_settings': data['kpt_dict']})
                    if data['potcar_mod'] is not None:
                        in_dict.update({'user_potcar_settings': data['potcar_mod']})
                    in_dict.update({'user_incar_settings': data['incar_mod']})
                    my_set = incar_set.__class__.from_dict(in_dict)
                    dft_pad = LaunchPad().from_dict(data['dft_pad'])
                    defuse_children = data['defuse_children']
                    my_wf = Workflow([SinglePointFW(structure=structure, vasp_input_set=my_set,
                                                    vasp_cmd=data['vasp_cmd'],
                                                    name='requested DFT {}'.format(data['wf_name']))],
                                     name='static request {}'.format(data['wf_name']),
                                     metadata={'added': 'requested DFT {}'.format(data['wf_name'])})
                    # print('ADDED STATIC')
                    dft_pad.add_wf(my_wf)
                    # return FWAction(defuse_children=False)  # , additions=[add_wf])

                elif data['symptom'] == 'window':
                    pass
                    # structure = Structure.from_dict(data['struct'])
                    # incar_set = MPStaticSet(structure)
                    # in_dict = incar_set.as_dict()
                    # kpts = Kpoints.automatic_gamma_density(structure, kppa=1000)
                    # in_dict.update({'user_kpoints_settings': kpts.as_dict()})
                    # in_dict.update({'user_incar_settings': data['incar_mod']})
                    # my_set = incar_set.__class__.from_dict(in_dict)
                    # dft_pad = LaunchPad().from_dict(data['dft_pad'])
                    # defuse_children = data['defuse_children']
                    # add_wf.append(Workflow([SinglePointFW(structure=structure, vasp_input_set=my_set,
                    #                                       vasp_cmd=data['vasp_cmd'],
                    #                                       name='energy window {}'.format(data['wf_name']))],
                    #                        metadata={'added': 'energy window {}'.format(data['wf_name'])}))
                    # add_stuff = True

                elif data['symptom'] == 'gradient':
                    structure = Structure.from_dict(data['struct'])
                    incar_set = MPStaticSet(structure)
                    in_dict = incar_set.as_dict()
                    in_dict.update({'user_kpoints_settings': data['kpt_dict']})
                    if data['potcar_mod'] is not None:
                        in_dict.update({'user_potcar_settings': data['potcar_mod']})
                    in_dict.update({'user_incar_settings': data['incar_mod']})
                    my_set = incar_set.__class__.from_dict(in_dict)
                    dft_pad = LaunchPad().from_dict(data['dft_pad'])
                    defuse_children = data['defuse_children']
                    add_wf.append(Workflow([SinglePointFW(structure=structure, vasp_input_set=my_set,
                                                          vasp_cmd=data['vasp_cmd'],
                                                          name='unsteady gradient {}'.format(data['wf_name']))],
                                           metadata={'added': 'unsteady gradient {}'.format(data['wf_name'])}))
                    add_stuff = True

                elif data['symptom'] == 'converge_flow':
                    structure = Structure.from_dict(data['struct'])
                    incar_set = MPStaticSet(structure)
                    in_dict = incar_set.as_dict()
                    in_dict.update({'user_kpoints_settings': data['kpt_dict']})
                    if data['potcar_mod'] is not None:
                        in_dict.update({'user_potcar_settings': data['potcar_mod']})
                    in_dict.update({'user_incar_settings': data['incar_mod']})
                    my_set = incar_set.__class__.from_dict(in_dict)
                    dft_pad = LaunchPad().from_dict(data['dft_pad'])
                    defuse_children = data['defuse_children']
                    add_wf.append(Workflow([SinglePointFW(structure=structure, vasp_input_set=my_set,
                                                          vasp_cmd=data['vasp_cmd'],
                                                          name='requested DFT {}'.format(data['wf_name']))],
                                           name='static request {}'.format(data['wf_name']),
                                           metadata={'added': 'requested DFT {}'.format(data['wf_name'])}))
                    add_stuff = True
                    # print('ADDED STATIC')
                    # dft_pad.add_wf(my_wf)

                else:
                    raise NotImplementedError('Wrong symptom')

            if add_stuff:
                # print('adding stuff')
                dft_pad.bulk_add_wfs(add_wf)
                return FWAction(defuse_children=defuse_children)  # ugly unsteady workaround


@explicit_serialize
class TGapSequential(TGapBase):
    """
    Class to run turbogap
    """
    required_params = ['params']
    optional_params = []

    def get_job(self, fw_spec):
        return TGapSequentialJob(params=self['params'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


@explicit_serialize
class TGapConverge(TGapBase):
    """
    Class to run turbogap
    """
    required_params = ['params']
    optional_params = []

    def get_job(self, fw_spec):
        return TGapConvergeJob(params=self['params'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


@explicit_serialize
class TGapPhaseDiagram(TGapBase):
    """
    Class to run turbogap for phase diagrams
    """
    required_params = ['params']
    optional_params = []

    def get_job(self, fw_spec):
        return TGapPhaseDiagramJob(params=self['params'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


class LammpsCGBase(FiretaskBase):
    """
    Base class to run LAMMPS, for inheritance
    """
    @abc.abstractmethod
    def get_job(self, fw_spec):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job(fw_spec)
        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        c.run()


@explicit_serialize
class LammpsCG(LammpsCGBase):
    """
    Class to run LAMMPS
    """
    required_params = ['lammps_params', 'db_info']
    optional_params = []

    def get_job(self, fw_spec):
        return LammpsCGJob(lammps_params=self['lammps_params'], db_info=self['db_info'], fw_spec=fw_spec)

    def get_validators(self):
        pass

    def get_handlers(self):
        return []

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)


class AimsBase(FiretaskBase):
    """
    Base class to run FHIaims, for inheritance
    """
    @abc.abstractmethod
    def get_job(self, fw_spec):
        pass

    @abc.abstractmethod
    def get_handlers(self):
        pass

    @abc.abstractmethod
    def get_validators(self):
        pass

    def run_task(self, fw_spec):
        job = self.get_job(fw_spec)
        c = Custodian(handlers=self.get_handlers(), jobs=[job], validators=self.get_validators(), max_errors=3)
        c.run()
        structure, params = job.get_relaxed_structure()
        if structure is not None:
            add_wf = Workflow([Firework(
                [Aims(aims_cmd=params['aims_cmd'], control=params['control'], structure=structure, basis_set='tight',
                      basis_dir=params['basis_dir'], single_basis=True, rerun_metadata=params['metadata'])])],
                metadata=params['metadata'], name='automatic tight run')
            return FWAction(additions=add_wf)


@explicit_serialize
class Aims(AimsBase):
    """
    Class to run FHIaims
    """
    required_params = ['aims_cmd', 'control', 'structure', 'basis_set', 'basis_dir', 'rerun_metadata', 'single_basis']
    optional_params = ['output_file', 'stderr_file']

    def get_job(self, fw_spec):
        raise NotImplementedError()
        # return AimsJob(aims_cmd=self['aims_cmd'], control=self['control'], structure=self['structure'],
        #                basis_set=self['basis_set'], basis_dir=self['basis_dir'], metadata=self['rerun_metadata'],
        #                single_basis=self['single_basis'])

    def get_validators(self):
        raise NotImplementedError()
        # return [AimsConvergedValidator()]

    def get_handlers(self):
        raise NotImplementedError()
        # return [AimsRelaxHandler(), FrozenJobErrorHandler()]

    def run_task(self, fw_spec):
        return super().run_task(fw_spec=fw_spec)
