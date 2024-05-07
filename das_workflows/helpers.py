import subprocess


def get_answer():
    """Get an answer."""
    return True


def env_chk(val, fw_spec, strict=True, default=None):
    """
    Gets what you need from the fw_spec env

    Args:
        val: what you are looking for
        fw_spec: the env
        strict: be strict
        default: default

    Returns:
        the val you are looking for
    """
    if val is None:
        return default

    if isinstance(val, str) and val.startswith(">>") and val.endswith("<<"):
        if strict:
            return fw_spec['_fw_env'][val[2:-2]]
        return fw_spec.get('_fw_env', {}).get(val[2:-2], default)
    return val


def find_binary(binary):
    """
    Finds an executable in bash shells

    Args:
        binary: name of the binary

    Returns:
        full path of the binary if it exists, None otherwise
    """

    try:
        result = subprocess.check_output(['which', str(binary)], encoding='utf8')
    except subprocess.CalledProcessError:
        result = None
    if result is None:
        raise FileNotFoundError('program >{}< not found in path'.format(binary))
    return result
