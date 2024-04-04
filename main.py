import os
import hydra

import pandas as pd
import numpy as np
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='.', config_name='config_upload')
def main(config: DictConfig) -> None:
   pass


if __name__ == '__main__':
    main()
