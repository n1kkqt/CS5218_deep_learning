import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    strides_from_shape,
    # MAX_DIMS,
)

import builtins


def tensor_map(fn):
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        out_index = np.empty(len(out_shape))
        in_index = np.empty(len(in_shape))
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_pos = index_to_position(in_index, in_strides)
            out[i] = fn(in_storage[in_pos])

    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.


    Returns:
        ret (:class:`TensorData`) : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        """
        Args:
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in, should broadcast with `a`

        Returns:
            return out
        """
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def sizeof(shape):
    size = 1
    for el in shape:
        size *= el
    return size


def get_index_normally(ordinal, shape):

    frac = ordinal
    out_index = len(shape) * [0]
    for i in range(len(shape)):
        frac, rem = frac // shape[-(i + 1)], frac % shape[-(i + 1)]
        out_index[-(i + 1)] = rem

    return out_index


def apply_fn_reduce(fn, ls):
    cur_el = ls[0]
    for i in range(1, len(ls)):
        cur_el = fn(cur_el, ls[i])
    return cur_el


def tensor_zip(fn):
    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        out_shape = shape_broadcast(a_shape, b_shape)
        out_strides[:] = strides_from_shape(out_shape)
        out_index = [0] * len(out_shape)

        a_index, b_index = [0] * len(a_shape), [0] * len(b_shape)
        a_broad_storage, b_broad_storage = [], []

        for i in range(sizeof(out_shape)):
            to_index(i, out_shape, out_index)

            broadcast_index(out_index, out_shape, a_shape, a_index)
            a_pos = index_to_position(a_index, a_strides)

            broadcast_index(out_index, out_shape, b_shape, b_index)
            b_pos = index_to_position(b_index, b_strides)

            a_broad_storage += [a_storage[a_pos]]
            b_broad_storage += [b_storage[b_pos]]

        out[:] = list(
            builtins.map(lambda x, y: fn(x, y), a_broad_storage, b_broad_storage)
        )

    return _zip


def zip(fn):

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):

        out_shape = (
            np.delete(out_shape, reduce_dim) if len(out_shape) > 1 else out_shape
        )
        out_strides = len(out_shape) * [0]
        out_strides[:] = (
            strides_from_shape(out_shape) if len(out_shape) > 0 else out_strides
        )

        out_indices = [
            get_index_normally(i, out_shape) for i in range(sizeof(out_shape))
        ]
        indices_for_fn = list(
            builtins.map(
                lambda ls: [
                    ls[:reduce_dim] + [i] + ls[reduce_dim:]
                    for i in range(a_shape[reduce_dim])
                ],
                out_indices,
            )
        )
        storage_indices_for_fn = list(
            builtins.map(
                lambda ls: list(
                    builtins.map(lambda index: index_to_position(index, a_strides), ls)
                ),
                indices_for_fn,
            )
        )
        storage_values_for_fn = list(
            builtins.map(
                lambda ls: list(builtins.map(lambda x: a_storage[x], ls)),
                storage_indices_for_fn,
            )
        )
        out[:] = list(
            builtins.map(lambda ls: apply_fn_reduce(fn, ls), storage_values_for_fn)
        )

        out = out.astype(float)
        # out_strides = np.array(out_strides).astype(int)

    return _reduce


def reduce(fn, start=0.0):

    f = tensor_reduce(fn)

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
