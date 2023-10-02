import os
import json
from typing import Dict, Any
import logging
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)

def overwrite_cfg(cfg: Dict, key: str, value: Any, recursive=False, check_exists=True):
    if check_exists:
        assert key in cfg, key
    if key in cfg and recursive and isinstance(value, dict):
        for k, v in value.items():
            overwrite_cfg(cfg[key], k, v, recursive=recursive, check_exists=check_exists)
    else:
        logger.info(f'overwrite key {key}: {cfg.get(key)} -> {value}')
        cfg[key] = value
    return cfg

def overwrite_cfg_from_dotlist(cfg, opts, write_env_vars=True):
    if len(opts) > 0:
        opts = OmegaConf.to_container(OmegaConf.from_dotlist(opts))
        for k, v in opts.items():
            overwrite_cfg(cfg, k, v, recursive=True)

        if write_env_vars and '_ENV_VARS_' in opts:
            # set to true only in the first call
            # otherwise env vars from base will be overwritten
            env_vars = opts['_ENV_VARS_']
            for k, v in env_vars.items():
                v = str(v)
                logger.info(f'set environ: {k}: {os.getenv(k)} -> {v}')
                os.environ[k] = v


def update_cfg_slurm(cfg, log_dir):
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_job_name = os.environ.get('SLURM_JOB_NAME')
    job_info_dir = 'experiments/slurm_id_to_out_dir'
    if slurm_job_id is not None:
        job_info_path = os.path.join(job_info_dir, f'{slurm_job_id}.json')
        if os.path.exists(job_info_path):
            assert slurm_job_name == 'bash', slurm_job_name
            logger.warning('should only happen in interactive session')
            count = 1
            while os.path.exists(job_info_path):
                job_info_path = os.path.join(job_info_dir, f'{slurm_job_id}_{count}.json')
                count += 1
        with open(job_info_path, 'w') as f:
            json.dump({'out_dir': os.path.abspath(log_dir)}, f)

    assert 'runtime' not in cfg, cfg['runtime']
    overwrite_cfg(cfg, 'runtime', {
        'slurm_job_id': slurm_job_id,
        'slurm_job_name': slurm_job_name,
    }, check_exists=False)


def resolve_with_omegaconf(config):
    try:
        OmegaConf.register_new_resolver("div", lambda x, y: x / y)
        OmegaConf.register_new_resolver("sub", lambda x, y: x - y)
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)
        OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
        OmegaConf.register_new_resolver("int", lambda x: int(x))
        def not_fn(x):
            assert isinstance(x, bool), x
            return not x
        OmegaConf.register_new_resolver("not", not_fn)
        OmegaConf.register_new_resolver("isnan", lambda x: x is None)
    except ValueError:
        # already registered
        pass

    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    return config


def load_cfg_from_path(path, **kwargs) -> Dict:
    path = os.path.abspath(path)
    if path.endswith('.yaml'):
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            cfg = json.load(f)
    else:
        raise NotImplementedError(path)
    cfg = load_cfg_from_dict(cfg, **kwargs)
    return cfg


def load_cfg_from_dict(cfg, allow_new_key=False, write_env_vars=True) -> Dict:
    if '__allow_new_key__' in cfg:
        allow_new_key = allow_new_key or cfg.pop('__allow_new_key__')
    if '_BASE_' in cfg:
        if cfg['_BASE_'] is not None and cfg['_BASE_'].startswith('$'):
            # temporarily resolve cfg to determine _BASE_
            cfg_base_path = resolve_with_omegaconf(cfg)['_BASE_']
        else:
            cfg_base_path = cfg['_BASE_']
        _ = cfg.pop('_BASE_')
        if cfg_base_path is not None:
            cfg_base = load_cfg_from_path(cfg_base_path, allow_new_key=allow_new_key, write_env_vars=False)  # recursive
            if 'runtime' in cfg_base:
                _ = cfg_base.pop('runtime')
            cfg, _ = update_recursive(cfg_base, cfg, allow_new_key=allow_new_key)

    if write_env_vars and '_ENV_VARS_' in cfg:
        # set to true only in the first call
        # otherwise env vars from base will be overwritten
        env_vars = cfg['_ENV_VARS_']
        for k, v in env_vars.items():
            v = str(v)
            logger.info(f'set environ: {k}: {os.getenv(k)} -> {v}')
            os.environ[k] = v
    return cfg
