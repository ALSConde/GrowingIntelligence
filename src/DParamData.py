from typing import Callable, Optional, Union

from sympy import expand
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
        assert len(new_shape) == len(self.shape), "Cannot add new dimensions."

        expands = [
            i
            for i, (snew, sold) in enumerate(zip(new_shape, self.shape))
            if snew > sold
        ]

        for i, (snew, sold) in enumerate(zip(new_shape, self.shape)):
            assert snew >= sold, f"Shape cannot decrease (dim {i}: {sold} → {snew})"

        if not expands:
            return self.data

        old_data = self._data.clone()
        old_shape = list(self.shape)
        self.reset_like(new_shape, init_function=padding_fn)

        copy_idx = [slice(s) for s in old_shape]
        self._data[tuple(copy_idx)] = old_data
        return self.data

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
