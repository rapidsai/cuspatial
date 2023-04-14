import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core._column.geometa import Feature_Enum


class GeoColumnBinopDispatch:
    """Dispatch binary operations of two geocolumns to subkernels and
    reassemble data.
    """

    def __init__(self, lhs: GeoColumn, rhs: GeoColumn):
        if len(lhs) != len(rhs):
            raise ValueError("Input is of different length.")

        self.lhs = lhs
        self.rhs = rhs
        self.dispatch_dict = None

    def __call__(self):
        if self.dispatch_dict is None:
            raise NotImplementedError(
                "Binary op dispatch base class cannot be used directly, use a "
                "derived class instead."
            )
        self._sort_geocolumn_by_type_pairs()
        self._compute_gather_offsets()
        self._gather_and_dispatch_computation()
        self._assemble_results()
        return self._reordered_result

    def _sort_geocolumn_by_type_pairs(self):
        """Given two geocolumn lhs and rhs, sort the offset buffer using
        the pair of types buffer column as keys.

        This result in a reordered view into child column where each subtype
        is contiguous in the offset buffer.
        """

        df = cudf.DataFrame(
            {
                "lhs_types": self.lhs._meta.input_types,
                "rhs_types": self.rhs._meta.input_types,
                "lhs_offsets": self.lhs._meta.union_offsets,
                "rhs_offsets": self.rhs._meta.union_offsets,
                "order": cp.arange(len(self.lhs)),
            }
        )

        self.df_sorted = df.sort_values(by=["lhs_types", "rhs_types"])

    def _compute_gather_offsets(self):
        """From the sorted types buffer and offset buffer, compute the
        boundaries of each contiguous section.
        """
        self.binary_type_to_offsets = {}
        self.orders = []
        for lhs_type, rhs_type in self.dispatch_dict:
            mask = (
                self.df_sorted["lhs_types"]
                == lhs_type.value & self.df_sorted["rhs_types"]
                == rhs_type.value
            )
            masked_df = self.df_sorted["lhs_offsets", "rhs_offsets", "order"][
                mask
            ]
            self.binary_type_to_offsets[(lhs_type, rhs_type)] = (
                masked_df["lhs_offsets"],
                masked_df["rhs_offsets"],
            )
            self.types_to_order[(lhs_type, rhs_type)] = masked_df["order"]

    def _gather_and_dispatch_computation(self):
        """For each type combination, gather from the child columns and
        dispatch to child kernel.
        """
        self.results = []
        for lhs_type, rhs_type in self.dispatch_dict:
            op, reflect = self.dispatch_dict[(lhs_type, rhs_type)]
            lhs_offsets, rhs_offsets = self.binary_type_to_offsets[
                (lhs_type, rhs_type)
            ]
            if op == "Impossible":
                self.result[(lhs_type, rhs_type)] = cp.full(
                    (len(lhs_offsets)), float("nan"), "f8"
                )
            else:
                lhs = self._gather_by_type(self.lhs, lhs_type, lhs_offsets)
                rhs = self._gather_by_type(self.rhs, rhs_type, rhs_offsets)
                self.results.append(
                    op(lhs, rhs) if not reflect else op(rhs, lhs)
                )

    def _assemble_results(self):
        """After computed all results from each subkernel, concatenate the
        result and make sure that the order matches that of original input.
        """
        if len(self.results) == 0:
            self.reordered_results = cudf.Series()

        results_concat = cudf.concat(self.results)
        orders_concat = cudf.concat(self.orders)
        self.reordered_results = cudf.Series(cp.zeros(results_concat))
        self.reordered_results[orders_concat] = results_concat

    def _gather_by_type(self, column, typ, offsets):
        if typ == Feature_Enum.POINT:
            return column.points._column.take(offsets)
        elif typ == Feature_Enum.MULTIPOINT:
            return column.mpoints._column.take(offsets)
        elif typ == Feature_Enum.LINESTRING:
            return column.lines._column.take(offsets)
        elif typ == Feature_Enum.POLYGON:
            return column.polygons._column.take(offsets)
        else:
            raise ValueError(f"Unrecognized type enum: {typ}")
