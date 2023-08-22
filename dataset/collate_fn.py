import numpy as np
import torch


def convert_to_dictofndarray(inputs, key_list):
    dict_of_ndarray = {}
    tmp = np.array(inputs)
    for i, key in enumerate(key_list):
        dict_of_ndarray[key] = tmp[:, i]
    return dict_of_ndarray


def standard_collate_fn(interaction_batch, key_list):
    interaction_batch = convert_to_dictofndarray(interaction_batch, key_list)

    for key, interaction_val in interaction_batch.items():
        interaction_batch[key] = torch.from_numpy(interaction_val)
    return interaction_batch


def snapshot_collate_fn(batch, key_list):
    interaction_batch = standard_collate_fn(batch, key_list)
    return interaction_batch
