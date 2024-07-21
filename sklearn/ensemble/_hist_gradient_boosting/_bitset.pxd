from ..utils._typedefs cimport uint8_t, uint32_t
from .common cimport X_BINNED_DTYPE_C
from .common cimport BITSET_DTYPE_C
from .common cimport BITSET_INNER_DTYPE_C
from .common cimport X_DTYPE_C

cdef void init_bitset(BITSET_DTYPE_C bitset) noexcept nogil

cdef void set_bitset(BITSET_DTYPE_C bitset, X_BINNED_DTYPE_C val) noexcept nogil

cdef uint8_t in_bitset(BITSET_DTYPE_C bitset, X_BINNED_DTYPE_C val) noexcept nogil

cpdef uint8_t in_bitset_memoryview(const BITSET_INNER_DTYPE_C[:] bitset,
                                         X_BINNED_DTYPE_C val) noexcept nogil

cdef uint8_t in_bitset_2d_memoryview(
    const BITSET_INNER_DTYPE_C [:, :] bitset,
    X_BINNED_DTYPE_C val,
    uint32_t row) noexcept nogil
