import argparse
import numpy as np
from arch_search.arch_search_densenet_net2net import arch_search_densenet
from arch_search.arch_search_convnet_net2net import arch_search_convnet

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

_SEED = 110
np.random.seed(_SEED)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--setting', type=str, default='convnet', choices=['convnet', 'densenet'],
)

args = parser.parse_args()
if args.setting == 'convnet':
    """
    Architecture Search on Convnet
    """
    arch_search_convnet(
        start_net_path='../start_nets/start_net_convnet_C10+_normal',
        arch_search_folder='../arch_search/Convnet/C10+/Conv_C10+_normal_acer_v',
        net_pool_folder='../net_pool/Convnet/C10+/Conv_C10+_normal_acer_v',
        max_episodes=30,
        random=False,
        baseline=True,
        acer=True,
    )
elif args.setting == 'densenet':
    """
    Architecture Search on DenseNet
    """
    arch_search_densenet(
        start_net_path='placeholder',
        arch_search_folder='placeholder',
        net_pool_folder='placeholder',
        max_episodes=15,
    )
else:
    pass
