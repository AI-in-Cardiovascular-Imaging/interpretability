import os
import pickle
import hydra

import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from utils.interpretability import Interpretability


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig) -> None:
   interpreter = Interpretability(config)
   models_dir = config.models_dir
   subdirs = [x[0] for x in os.walk(models_dir)][1:]  # skip the first element which is the root directory

   for dir in subdirs:
      logger.info(f'Processing {dir}')
      train = pd.read_csv(os.path.join(dir, 'train_feature.csv'))
      test = pd.read_csv(os.path.join(dir, 'test_feature.csv'))
      with open(os.path.join(dir, 'model.pickle'), 'rb') as f:
         model = pickle.load(f)

      interpreter(train, test, model)


if __name__ == '__main__':
    main()
