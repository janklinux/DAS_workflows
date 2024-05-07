import datetime
from fireworks import Firework, Workflow
from das_workflows.firetasks.base import Lammps, TGapSequential, TGapConverge
from das_workflows.firetasks.vasp import SinglePointFW
from atomate.vasp.fireworks.core import OptimizeFW


def run_lammps(lammps_params, structure, parents):
    """
    Runs LAMMPS with the supplied structures
    Use when a trained potential exists and we need more LAMMPS runs with it

    Args:
        lammps_params:  parameters for LAMMPS
        structure: pymatgen structure
        parents: the parents of this run
    Returns:
        the firework for assembly
    """

    params = lammps_params.copy()
    params['structure'] = structure.as_dict()

    return Workflow([Firework([Lammps(lammps_params=params)], name='LAMMPS_FW', parents=parents)], name='lammps')


def vasp_static_wf(structure, struc_name='', name='Static_run', vasp_input_set=None,
                   vasp_cmd=None, user_kpoints_settings=None, tag=None, metadata=None):
    """
    Static VASP workflow, to generate single point DFT training data
    Args:
        structure: pymatgen structure object
        struc_name: name of the structure
        name: name of the workflow
        vasp_input_set: materials project input set
        vasp_cmd: command to run
        user_kpoints_settings: kpoints object for k-grid
        tag: tag for the wokflow
        metadata: additional data for the workflow

    Returns:
        the workflow add into LaunchPad
    """
    if vasp_input_set is None:
        raise ValueError('INPUTSET needs to be defined...')
    if user_kpoints_settings is None:
        raise ValueError('You have to specify the K-grid...')
    if vasp_cmd is None:
        raise ValueError('vasp_cmd needs to be set by user...')
    if tag is None:
        tag = datetime.datetime.now().strftime('%Y/%m/%d-%T')

    vis = vasp_input_set
    v = vis.as_dict()
    v.update({"user_kpoints_settings": user_kpoints_settings})
    vis_static = vis.__class__.from_dict(v)

    fws = [SinglePointFW(structure=structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd,
                         name="{} -- static".format(tag))]

    wfname = "{}: {}".format(struc_name, name)
    return Workflow(fws, name=wfname, metadata=metadata)


def vasp_relax_wf(structure, struc_name='', name='Relax_run', vasp_input_set=None,
                  vasp_cmd=None, db_file=None, user_kpoints_settings=None, tag=None, metadata=None):

    if vasp_input_set is None:
        raise ValueError('INPUTSET needs to be defined...')
    if user_kpoints_settings is None:
        raise ValueError('You have to specify the K-grid...')
    if vasp_cmd is None:
        raise ValueError('vasp_cmd needs to be set by user...')
    if tag is None:
        tag = datetime.datetime.now().strftime('%Y/%m/%d-%T')

    vis = vasp_input_set
    v = vis.as_dict()
    v.update({"user_kpoints_settings": user_kpoints_settings})
    vis_static = vis.__class__.from_dict(v)

    fws = [OptimizeFW(structure=structure, vasp_input_set=vis_static, vasp_cmd=vasp_cmd, job_type='full_opt_run',
                      db_file=db_file, name="{} -- relax".format(tag))]
    wfname = "{}: {}".format(struc_name, name)

    return Workflow(fws, name=wfname, metadata=metadata)


def tgap_sequential(params, parents, my_fw_spec=None):
    """
    Skeleton to build a sequential turbogap runs
    Args:
        params(dict): all parameters for the turbogap run
        parents: the possible parents
        my_fw_spec: user defined spec modifications
    Returns:
        firework to build dependent workflow
    """

    local_params = params.copy()
    if my_fw_spec is not None:
        local_spec = my_fw_spec
        return Firework([TGapSequential(params=local_params)], name='TGap Sequence',
                        parents=parents, spec=local_spec)
    else:
        return Firework([TGapSequential(params=local_params)], name='TGap Sequence',
                        parents=parents)


def tgap_converge(params, parents):
    """
    Skeleton to build a phase diagram using turbogap for the given input structure
    Args:
        params(dict): all parameters for the turbogap run
        parents: the possible parents
    Returns:
        firework to build dependent workflow
    """
    local_params = params.copy()
    return Firework([TGapConverge(params=local_params)], name='TGap multi-step relaxation', parents=parents)
