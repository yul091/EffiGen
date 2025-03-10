
from typing import Dict, Union, Any
from collections.abc import Mapping
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import DynamicCache



# Custom padding function
def pad_batch(tensor_list, pad_value=0):
    """
    Pads a list of tensors to the same length along dimension 1.
    """
    return pad_sequence(tensor_list, batch_first=True, padding_value=pad_value)


def prepare_decoding_inputs(
    inputs: Dict[str, Union[torch.Tensor, Any]],
):
    # Create a new dictionary to avoid modifying the original inputs
    new_inputs = {key: val.clone() if torch.is_tensor(val) else val for key, val in inputs.items()}
    labels = new_inputs.pop("labels", None)
    if labels is not None:
        labels_attention_mask = torch.ones_like(labels)
        new_inputs['input_ids'] = torch.cat((inputs['input_ids'], labels), dim=1)
        new_inputs['attention_mask'] = torch.cat((inputs['attention_mask'], labels_attention_mask), dim=1)
        new_labels = torch.cat(
            (-100 * torch.ones_like(inputs['input_ids']), labels), dim=1
        )
        new_inputs['labels'] = new_labels
    return new_inputs


def _prepare_input(
    data: Union[torch.Tensor, Any],
    device: torch.device = 'cuda',
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, DynamicCache):
        data.key_cache = _prepare_input(data.key_cache, device)
        data.value_cache = _prepare_input(data.value_cache, device)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    return data
    

def prepare_inputs(
    inputs: Dict[str, Union[torch.Tensor, Any]],
    device: torch.device = 'cuda',
) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    new_inputs = _prepare_input(inputs, device=device)
    if new_inputs is None or len(new_inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it."
        )
    return new_inputs