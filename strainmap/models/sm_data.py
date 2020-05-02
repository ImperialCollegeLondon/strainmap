from __future__ import annotations

from typing import Union, Dict, Sequence, Tuple, Optional, Callable, List
from copy import deepcopy
from enum import Enum, auto
from dataclasses import dataclass
from itertools import chain

import numpy as np
import sparse


class Region(Enum):
    GLOBAL = auto()
    ANGULAR = auto()
    RADIAL = auto()


class Background(Enum):
    NONE = auto()
    ESTIMATED = auto()
    PHANTOM = auto()


@dataclass(frozen=True)
class RegionalLabel:
    """Hashable regional labels."""

    kind: Region
    regions: int
    bg: Background = Background.ESTIMATED

    @classmethod
    def from_str(cls, string: str):
        split = string.split("-")
        return cls(
            Region[split[0].upper()], int(split[1]), Background[split[2].upper()]
        )

    @property
    def as_str(self):
        return f"{self.kind.name.lower()}-{self.regions}-{self.bg.name.lower()}"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.as_str == other
        elif isinstance(other, RegionalLabel):
            return (
                self.kind == other.kind
                and self.regions == other.regions
                and self.bg == other.bg
            )
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.as_str)


class LabelledArray:
    """ Provides dimensions and coordinates accessibility by labels.

    Provides numpy and sparse.COO arrays with a thin layer to access dimensions and
    coordinates by label rather in addition to indices. It also incorporates some
    convenience methods like sum, cumsum or stack using labels rather than indices.
    """

    def __init__(
        self,
        dims: Sequence[str],
        coords: Dict[str, Sequence],
        values: Union[np.ndarray, sparse.DOK, sparse.COO],
    ):

        if len(dims) != len(values.shape):
            raise RuntimeError(
                f"The numer of labels for the dimensions ({len(dims)}) is different"
                f"than the number of dimensions in the array ({len(values.shape)})."
            )

        for k, v in coords.items():
            if k not in dims:
                raise RuntimeError(f"Unknown dimension {k}.")
            elif v is not None and len(v) != values.shape[dims.index(k)]:
                raise RuntimeError(
                    f"Number of coordinates ({len(v)}) do not match the number of"
                    f"elements in dimension '{k}' ({values.shape[dims.index(k)]})."
                )

        self.dims = tuple(dims)
        self.coords = {d: coords.get(d, None) for d in dims}
        self.values = values
        self.module = __import__(type(values).__module__.split(".")[0])

    def __getitem__(self, items) -> Union[LabelledArray, np.ndarray, sparse.COO]:
        """ Gets items from 'values' using indices.

        If the result is an scalar, then this function returns an scalar, otherwise
        it will return a LabelArray object with the corresponding dimensions and
        coordinates.
        """
        assert not any(
            isinstance(item, type(Ellipsis)) for item in items
        ), NotImplementedError

        new_value = self.values[items]
        new_dims = [
            d
            for i, d in enumerate(self.dims)
            if i >= len(items) or not isinstance(items[i], int)
        ]

        if len(new_dims) == 0:
            return new_value

        new_coords: Dict[str, Optional[Sequence[str]]] = {}
        for i, d in enumerate(self.dims):
            if d not in new_dims:
                continue
            elif self.coords[d] is None:
                new_coords[d] = None
            elif i >= len(items):
                new_coords[d] = self.coords[d]
            else:
                new_coords[d] = self.coords[d][items[i]]

        return LabelledArray(new_dims, new_coords, new_value)

    def __str__(self) -> str:
        dim_str = ", ".join(
            [f"'{d}': {self.values.shape[i]}" for i, d in enumerate(self.dims)]
        )
        coord_str = "\n".join([f"- '{d}': {v}" for d, v in self.coords.items()])
        return "Dims:\n- {}\n\nCoords:\n{}\n\nValues: {}\n{}\n".format(
            dim_str, coord_str, type(self.values), self.values
        )

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: object):
        if not isinstance(other, LabelledArray):
            return NotImplemented
        return (
            self.dims == other.dims
            and self.coords == other.coords
            and (self.values == other.values).all()
        )

    @property
    def shape(self) -> Tuple[int]:
        return self.values.shape

    def sel(self, **kwargs) -> Union[LabelledArray, np.ndarray, sparse.COO]:
        """ Gets items from 'values' using dimension and coordinate labels.

        For coordinates without labels or if the dimension name is followed by
        '__i' (double undescore + i) then regular indices are used for that dimension.
        """
        keys = (self._process_keys(k, v) for k, v in kwargs.items())
        filled: List[Union[int, slice]] = [slice(None)] * len(self.dims)
        for k in keys:
            filled[k[0]] = k[1]
        return self[tuple(filled)]

    def _process_keys(
        self, dim: str, coord: Union[str, int, slice]
    ) -> Tuple[int, Union[int, slice]]:
        """ Transform a dimension and coordinate pair into indices.

        For coordinates, the indices can be an integer or a slice, if more than one
        element are equal to the coordinate string. In this case, it is assumed that
        the coordinates with the same labels are consecutive.
        """
        didx = self.dims.index(dim)
        cidx: Union[int, slice]
        if isinstance(coord, str):
            if self.coords[dim] is None:
                raise ValueError(f"Dimension '{dim}' has no labelled coordinates.")
            clist = [i for i, c in enumerate(self.coords[dim]) if c == coord]
            cidx = clist[0] if len(clist) == 1 else slice(clist[0], clist[-1] + 1)
        else:
            cidx = coord

        return didx, cidx

    def transpose(self, *dims: str) -> LabelledArray:
        """ Transposes the dimensions of the LabelledArray.

        This function returns a new LabelledArray with the dimensions transposed.
        """
        if set(dims) != set(self.dims):
            raise RuntimeError(
                "The transposed dimensions must match the existing dimentions."
            )

        axes = tuple((self.dims.index(d) for d in dims))
        new_values = self.values.transpose(axes)
        return LabelledArray(dims, self.coords, new_values)

    def _reduction(
        self, operation: Callable, dims: Optional[Sequence[str]] = None, **kwargs
    ) -> Union[int, "LabelledArray"]:
        """ Common method for dimension-reducing operations.
        """
        if dims is None:
            return operation(**kwargs)

        axis = tuple([self.dims.index(d) for d in dims])
        new_values = operation(axis=axis, **kwargs)
        new_dims = tuple([d for d in self.dims if d not in dims])
        new_coords = {d: c for d, c in self.coords.items() if d not in dims}

        return LabelledArray(new_dims, new_coords, new_values)

    def mean(
        self, dims: Optional[Sequence[str]] = None, **kwargs
    ) -> Union[int, "LabelledArray"]:
        """ Calculates the mean of the array across the chosen dimensions.
        """
        return self._reduction(self.values.mean, dims=dims, **kwargs)

    def sum(
        self, dims: Optional[Sequence[str]] = None, **kwargs
    ) -> Union[int, "LabelledArray"]:
        """ Calculates the sum of the array across the chosen dimensions.
        """
        return self._reduction(self.values.sum, dims=dims, **kwargs)

    def cumsum(
        self, dim: Optional[str] = None, **kwargs
    ) -> Union[int, "LabelledArray"]:
        """ Calculates the cumulative sum of the array along the given dimension.
        """
        if isinstance(self.values, sparse.COO):
            raise NotImplementedError("Cumsum is not implemented for COO arrays.")

        if dim is None:
            return np.cumsum(self.values, **kwargs)

        axis = self.dims.index(dim)
        new_values = np.cumsum(self.values, axis=axis, **kwargs)

        return LabelledArray(self.dims, deepcopy(self.coords), new_values)

    def concatenate(
        self, args: Sequence["LabelledArray"], dim: str, **kwargs
    ) -> Union[int, "LabelledArray"]:
        """ Concatenate any number of LabelledArrays along the given dimension.

        Coordinates along that dimension are joined together, so they must be of the
        same type (eg. all None or all Lists).
        """
        if dim not in self.dims or not all([dim in a.dims for a in args]):
            raise KeyError(f"Unknown dimension '{dim}' for one or more input arrays.")

        if not all([isinstance(a.coords[dim], type(self.coords[dim])) for a in args]):
            raise TypeError(
                f"The coodinates for dimension '{dim}' must have the same type in "
                "all input arrays."
            )

        new_values = self.module.concatenate(
            [self.values] + [a.values for a in args],
            axis=self.dims.index(dim),
            **kwargs,
        )
        new_coords = deepcopy(self.coords)
        if new_coords[dim] is not None:
            new_coords[dim] = list(
                chain(new_coords[dim], *[a.coords[dim] for a in args])
            )

        return LabelledArray(self.dims, new_coords, new_values)

    def stack(
        self,
        args: Sequence["LabelledArray"],
        dim: str,
        coords: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Union[int, "LabelledArray"]:
        """ Stacks any number of LabelledArrays along a new dimension.

        All LabelledArrays to be stacked must have the same dimensions coordinates and
        shapes. If provided, the coordinates for the new dimension must be a sequence
        with the length of the number of arrays to be stacked.
        """
        if not all([self.dims == a.dims for a in args]):
            raise ValueError("All arrays must have the same dimensions.")

        if not all([self.coords == a.coords for a in args]):
            raise ValueError("All arrays must have the same coordinates.")

        if coords is not None and len(coords) != len(args) + 1:
            raise ValueError(
                f"Parameter 'coords' must be None or a sequence of "
                f"length {len(args) + 1}."
            )

        new_values = self.module.stack(
            [self.values] + [a.values for a in args], axis=0, **kwargs
        )
        new_coords = deepcopy(self.coords)
        new_coords[dim] = coords
        new_dims = (dim,) + self.dims

        return LabelledArray(new_dims, new_coords, new_values)
