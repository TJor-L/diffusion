import os
# import pytorch_lightning as lightning
import yaml
import sys
from guided_diffusion.run import run as diffusion_run


# lightning.seed_everything(1016)

with open(sys.argv[1] if len(sys.argv) > 1 else 'configs/fastmri_config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)

if config['setting']['method'] != 'diffusion':
    os.environ["CUDA_VISIBLE_DEVICES"] = config['setting']['gpu_index']

METHOD_LIST = {
    'diffusion': diffusion_run,
}

# raise ValueError(f"config['setting']['method']: {config['setting']['method']}\nconfig: {config}")
METHOD_LIST[config['setting']['method']](config)
