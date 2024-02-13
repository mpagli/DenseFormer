import numpy as np
import torch

from . import pg19, openwebtext2

PREPARE_GET_DATASET_MAP = {
    "pg19": (pg19.prepare_pg19_data, pg19.get_pg19_data),
    "owt2": (openwebtext2.prepare_openwebtext2_data, openwebtext2.get_openwebtext2_data)
}


def prepare_dataset(args):
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own pythin file. The expected format at the moment is a disctionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    return PREPARE_GET_DATASET_MAP[args.dataset][0](args)

def get_dataset(args):
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own pythin file. The expected format at the moment is a disctionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    return PREPARE_GET_DATASET_MAP[args.dataset][1](args)
