from __future__ import annotations

import operator as op
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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

    @classmethod
    def from_dict(
        cls,
        dims: Sequence,
        values: Dict,
        coords: Optional[Dict[str, Sequence]] = None,
        skip: Tuple[str] = ("",),
    ) -> LabelledArray:
        """Transforms a (possibly nested) dictionary of np.ndarray in a LabelledArray.

         If the dictionary is nested, all branches must have the same depth and keys on
         each level. The arrays must have the same shape. dims must be a list with the
         number of elements equal to the depth of the dictionary + the shape of the
         array. If the number of non-zero elements is smaller than 75% of the total, a
         sparse.COO array is used.

        Args:
            dims: Sequence of dimension names.
            values: A nested dictionary with numpy arrays in the deepest level.
            coords: (Optional) Dictionary of coordinates for one or more dimensions.
            skip: (optional) Keys to skip when scanning the dictionaries.

        Returns:
            A LabelledArray object
        """

        def get_coords(d: Sequence, v: Union[Dict, np.ndarray], c: Dict) -> Dict:
            c[d[0]] = (
                [k for k in v.keys() if k not in skip] if isinstance(v, dict) else None
            )
            if len(d) > 1:
                return get_coords(
                    d[1:], v.get(c[d[0]][0]) if isinstance(v, dict) else v[0], c
                )
            return c

        new_coords = get_coords(dims, values, {})
        if coords is not None:
            new_coords.update(coords)

        def get_values(v: Union[Dict, np.ndarray]) -> np.ndarray:
            if isinstance(v, dict):
                return np.stack(
                    [get_values(val) for key, val in v.items() if key not in skip]
                )
            return v

        new_values = get_values(values)
        if np.count_nonzero(new_values) / new_values.size < 0.90:
            new_values = sparse.COO(new_values)

        return cls(dims=dims, coords=new_coords, values=new_values)

    def __init__(
        self,
        dims: Sequence[str],
        coords: Dict[str, Sequence],
        values: Union[np.ndarray, sparse.COO],
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

    def __getitem__(self, idx) -> Union[LabelledArray, np.ndarray, sparse.COO]:
        """ Gets items from 'values' using indices.

        If the result is an scalar, then this function returns an scalar, otherwise
        it will return a LabelArray object with the corresponding dimensions and
        coordinates.
        """
        items = (idx,) if not isinstance(idx, Sequence) else idx

        if any(isinstance(item, type(Ellipsis)) for item in items):
            raise NotImplementedError

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

    def __add__(self, other: object) -> LabelledArray:
        return self._operate(other, op.add)

    def __radd__(self, other: object) -> LabelledArray:
        return self._operate(other, op.add)

    def __mul__(self, other: object) -> LabelledArray:
        return self._operate(other, op.mul)

    def __rmul__(self, other: object) -> LabelledArray:
        return self._operate(other, op.mul)

    def _operate(self, other: object, operation: Callable) -> LabelledArray:
        """ Carries the actual mathematical operation.

        If the `other` argument is not a LabelledArray, the operation is transferred
        directly to an operation on the `self.values`, and therefore will fail if that
        operation makes no sense for whatever reason.

        Otherwise, some manipulation is necessary to ensure that the underlying values,
        which can be np.ndarray, sparse.COO or a mixture of both are compatible.

        Finally dimensions are aligned to perform the operation.

        Args:
            other: Other object to operate with this one.
            operation: Operation to be done.

        Returns:
            The result of the operation.
        """
        if not isinstance(other, LabelledArray):
            return LabelledArray(self.dims, self.coords, operation(self.values, other))

        if not isinstance(self.values, type(other.values)):
            left, right = self.match_type(other, operation)
            left, right = left.align(right)
        else:
            left, right = self.align(other)

        return LabelledArray(
            left.dims,
            {
                k: left.coords[k] if left.coords[k] is not None else right.coords[k]
                for k in left.dims
            },
            operation(left.values, right.values),
        )

    def to_coo(self) -> LabelledArray:
        """Transform a dense LabelledArray to its sparse version."""
        if isinstance(self.values, sparse.COO):
            return self
        return LabelledArray(self.dims, self.coords, sparse.COO(self.values))

    def to_dense(self) -> LabelledArray:
        """Transform a sparse LabelledArray to its dense version."""
        if isinstance(self.values, np.ndarray):
            return self
        return LabelledArray(self.dims, self.coords, self.values.todense())

    def match_type(
        self, other: LabelledArray, operation: Callable
    ) -> Tuple[LabelledArray, ...]:
        """  Matches the type of the arrays depending on the operation.

        Mixed operations between np.ndarray and COO are not really allowed and they
        need to be transformed to the same type.

        - For additions: the COO array is densified.
        - For multiplications: the np.ndarray is transformed to a COO array.

        Args:
            other: Other object to operate with this one.
            operation: Operation to be done.

        Returns:
            A tuple with the arrays with match type.
        """
        if operation is op.add:
            return (
                (self.to_dense(), other)
                if isinstance(self.values, sparse.COO)
                else (self, other.to_dense())
            )
        elif operation is op.mul:
            return (
                (self.to_coo(), other)
                if isinstance(self.values, np.ndarray)
                else (self, other.to_coo())
            )
        else:
            raise ValueError(f"Unsupported operation: {operation}.")

    @property
    def shape(self) -> Tuple[int]:
        return self.values.shape

    def len_of(self, dim: str) -> int:
        """ Returns the length of the LabelledArray in that dimension.

        Args:
            dim: Dimension name

        Returns:
            The length of the array along the given dimension.
        """
        return self.shape[self.dims.index(dim)]

    def sel(self, **kwargs) -> Union[LabelledArray, np.ndarray, sparse.COO]:
        """ Gets items from 'values' using dimension and coordinate labels.
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

    def align(self, right: LabelledArray) -> Tuple[LabelledArray, ...]:
        """ Align the dimensions of the LabelledArrays so they can be used in maths.

        Coordinates along the common dimensions must be identical.

        Args:
            right: The other LabelledArray to align with this one.

        Returns:
            A tuple with two LabelledArrays with aligned dimensions.
        """
        common_dims = tuple((d for d in self.dims if d in right.dims))
        dims = sorted(
            common_dims
            + tuple((d for d in self.dims + right.dims if d not in common_dims))
        )

        if not all(self.coords[k] == right.coords[k] for k in common_dims):
            raise ValueError("Common dimensions must have identical coordinates.")

        lvalues = self.transpose(*(d for d in dims if d in self.dims)).values[
            tuple(slice(None) if d in self.dims else None for d in dims)
        ]
        rvalues = right.transpose(*(d for d in dims if d in right.dims)).values[
            tuple(slice(None) if d in right.dims else None for d in dims)
        ]

        return (
            LabelledArray(dims, self.coords, lvalues),
            LabelledArray(dims, right.coords, rvalues),
        )

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
    ) -> Union[int, LabelledArray]:
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
    ) -> Union[int, LabelledArray]:
        """ Calculates the mean of the array across the chosen dimensions.
        """
        return self._reduction(self.values.mean, dims=dims, **kwargs)

    def sum(
        self, dims: Optional[Sequence[str]] = None, **kwargs
    ) -> Union[int, LabelledArray]:
        """ Calculates the sum of the array across the chosen dimensions.
        """
        return self._reduction(self.values.sum, dims=dims, **kwargs)

    def cumsum(self, dim: Optional[str] = None, **kwargs) -> Union[int, LabelledArray]:
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
        self, args: Sequence[LabelledArray], dim: str, **kwargs
    ) -> Union[int, LabelledArray]:
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
        args: Sequence[LabelledArray],
        dim: str,
        coords: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Union[int, LabelledArray]:
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
