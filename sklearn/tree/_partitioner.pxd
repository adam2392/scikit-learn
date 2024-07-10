# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# See _partitioner.pyx for details.
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int8_t, int32_t, uint32_t


# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7


cdef class Partitioner:
    cdef intp_t[::1] samples
    cdef float32_t[::1] feature_values
    cdef intp_t start
    cdef intp_t end
    cdef intp_t n_missing
    cdef const unsigned char[::1] missing_values_in_feature_mask

    cdef void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil
    cdef void init_node_split(
        self,
        intp_t start,
        intp_t end
    ) noexcept nogil
    cdef void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil
    cdef void next_p(
        self,
        intp_t* p_prev,
        intp_t* p
    ) noexcept nogil
    cdef intp_t partition_samples(
        self,
        float64_t current_threshold
    ) noexcept nogil
    cdef void partition_samples_final(
        self,
        intp_t best_pos,
        float64_t best_threshold,
        intp_t best_feature,
        intp_t n_missing,
    ) noexcept nogil


cdef class DensePartitioner(Partitioner):
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef const float32_t[:, :] X


cdef class SparsePartitioner(Partitioner):
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef const float32_t[::1] X_data
    cdef const int32_t[::1] X_indices
    cdef const int32_t[::1] X_indptr

    cdef intp_t n_total_samples

    cdef intp_t[::1] index_to_samples
    cdef intp_t[::1] sorted_samples

    cdef intp_t start_positive
    cdef intp_t end_negative
    cdef bint is_samples_sorted

    cdef void extract_nnz(
        self,
        intp_t feature
    ) noexcept nogil
    cdef intp_t _partition(
        self,
        float64_t threshold,
        intp_t zero_pos
    ) noexcept nogil
