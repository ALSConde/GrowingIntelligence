from typing import Callable, Optional, Union
import torch


class DParamData(object):
    def __init__(
        self,
        name: str,
        shape: Optional[tuple] = None,
        init_function: Callable[[torch.Size], torch.Tensor] = torch.zeros,
        init_tensor: Union[torch.Tensor, None] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        assert isinstance(name, str)
        assert (init_tensor is not None) or (shape is not None)
        if init_tensor is not None and shape is not None:
            assert init_tensor.shape == shape

        self.init_function = init_function
        self.name = name

        if shape is not None:
            self.shape = torch.Size(shape)
        else:
            assert init_tensor is not None
            self.shape = init_tensor.size()

        self.device = torch.device(device)
        if init_tensor is not None:
            self._data: torch.Tensor = init_tensor
        else:
            self.reset_like()

    def reset_like(self, shape=None, init_function=None):
        if shape is not None:
            self.shape = torch.Size(shape)
        if init_function is None:
            init_function = self.init_function
        self._data = init_function(self.shape).to(self.device)

    def expand(self, new_shape, padding_fn=torch.zeros):
        if isinstance(new_shape, int) and new_shape == self.shape:
            return self._data
        elif not isinstance(new_shape, int) and list(new_shape) == list(self.shape):
            return self._data

        new_tensor = padding_fn(new_shape, device=self._data.device)
        copy_idx = tuple(
            slice(0, min(s_old, s_new)) for s_old, s_new in zip(self.shape, new_shape)
        )
        new_tensor[copy_idx] = self._data[copy_idx]

        return new_tensor

    def expand_(self, new_shape, padding_fn=torch.zeros):
        expanded = self.expand(new_shape, padding_fn)
        self._data.resize_(expanded.shape)
        self._data.copy_(expanded)
        self.shape = torch.Size(expanded.shape)
        return self

    def prune(self, indices_to_remove, dim=0):
        if not isinstance(indices_to_remove, torch.Tensor):
            indices_to_remove = torch.tensor(
                indices_to_remove, dtype=torch.long, device=self._data.device
            )

        total_size = self._data.size(dim)
        keep_mask = torch.ones(total_size, dtype=torch.bool, device=self._data.device)
        keep_mask[indices_to_remove] = False

        keep_indices = torch.arange(total_size, device=self._data.device)[keep_mask]

        new_data = torch.index_select(self._data, dim, keep_indices)

        self._data = new_data
        self.shape = new_data.size()
        return self

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value):
        assert (
            value.shape == self._data.shape
        ), "Shape mismatch. Use expand/reset_like if shape changes."
        self._data = value

    def __str__(self):
        return f"DParamData_{self.name}:{self.shape}:{self._data}"
