# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cython cimport final
from libc.math cimport isnan
from libc.stdlib cimport qsort
from libc.string cimport memcpy

from ..utils._typedefs cimport int8_t
from ._criterion cimport Criterion
from ._utils cimport (RAND_R_MAX, bs_from_template, bs_get, bs_set, log,
                      rand_int, rand_uniform, setup_cat_cache)

import numpy as np
from scipy.sparse import issparse


cdef float64_t INFINITY = np.inf

# Allow for 32 bit float comparisons
cdef float32_t INFINITY_32t = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef float32_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparsePartitioner
cdef float32_t EXTRACT_NNZ_SWITCH = 0.1

cdef inline void _init_split(SplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.split_value.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0

cdef class Splitter:
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst,
        bint breiman_shortcut,
        *argv
    ):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : intp_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : intp_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : float64_t
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness

        monotonic_cst : const int8_t[:]
            Monotonicity constraints

        breiman_shortcut : bool
            Whether to use the breiman shortcut or not when possible.
        """

        self.criterion = criterion

        self.n_samples = 0
        self.n_features = 0

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.monotonic_cst = monotonic_cst
        self.with_monotonic_cst = monotonic_cst is not None
        self.breiman_shortcut = breiman_shortcut

        if self.breiman_shortcut:
            self.sort_value = np.zeros(64, dtype=np.float32)
            self.sort_density = np.zeros(64, dtype=np.float32)

            # XXX: unsure what this it.
            self.cat_offs = np.empty(64, dtype=np.int32)
            # A storage of the sorted categories used in Breiman shortcut
            self.sorted_cat = np.empty(64, dtype=np.intp)

        else:
            self.sort_value = np.zeros(1, dtype=np.float32)
            self.sort_density = np.zeros(1, dtype=np.float32)

            # XXX: unsure what this it.
            self.cat_offs = np.empty(1, dtype=np.int32)
            # A storage of the sorted categories used in Breiman shortcut
            self.sorted_cat = np.empty(1, dtype=np.intp)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __reduce__(self):
        return (type(self), (
            self.criterion,
            self.max_features,
            self.min_samples_leaf,
            self.min_weight_leaf,
            self.random_state,
            self.monotonic_cst,
            self.breiman_shortcut
        ), self.__getstate__())

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
        const int32_t[::1] n_categories,
    ) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : ndarray, dtype=float64_t
            This is the vector of targets, or true labels, for the samples represented
            as a Cython memoryview.

        sample_weight : ndarray, dtype=float64_t
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight. This is represented
            as a Cython memoryview.

        has_missing : bool
            At least one missing values is in X.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef intp_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        self.samples = np.empty(n_samples, dtype=np.intp)
        cdef intp_t[::1] samples = self.samples

        cdef intp_t i, j
        cdef float64_t weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight is None or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight is not None:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef intp_t n_features = X.shape[1]
        self.features = np.arange(n_features, dtype=np.intp)
        self.n_features = n_features

        self.feature_values = np.empty(n_samples, dtype=np.float32)
        self.constant_features = np.empty(n_features, dtype=np.intp)

        self.y = y

        self.sample_weight = sample_weight
        if missing_values_in_feature_mask is not None:
            self.criterion.init_sum_missing()

        # Initialize the number of categories for each feature
        # A value of -1 indicates a non-categorical feature
        if n_categories is None:
            self.n_categories = np.array([-1] * n_features, dtype=np.int32)
        else:
            self.n_categories = np.empty(n_categories, dtype=np.int32)
            self.n_categories[:] = n_categories

        # If needed, allocate cache space for categorical splits
        cdef int32_t max_n_categories = max(self.n_categories)
        if max_n_categories > 0:
            cache_size = (max_n_categories + 63) // 64
            self.cat_cache[:] = np.empty(cache_size, dtype=np.uint32)
        else:
            self.cat_cache[:] = np.empty(1, dtype=np.uint32)
        return 0

    cdef int node_reset(
        self,
        intp_t start,
        intp_t end,
        float64_t* weighted_n_node_samples
    ) except -1 nogil:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : intp_t
            The index of the first sample to consider
        end : intp_t
            The index of the last sample to consider
        weighted_n_node_samples : ndarray, dtype=float64_t pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(
            self.y,
            self.sample_weight,
            self.weighted_n_samples,
            self.samples,
            start,
            end
        )

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:

        """Find the best split on node samples[start:end].

        This is a placeholder method. The majority of computation will be done
        here.

        It should return -1 upon errors.
        """

        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""

        self.criterion.clip_node_value(dest, lower_bound, upper_bound)

    cdef float64_t node_impurity(self) noexcept nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()

    cdef inline void _breiman_sort_categories(
        self,
        intp_t start,
        intp_t end,
        int32_t ncat,
        intp_t ncat_present,
        const int32_t[:] cat_offset,
        intp_t[:] sorted_cat
    ) noexcept nogil:
        """The Breiman shortcut for finding the best split involves a
        preprocessing step wherein we sort the categories by
        increasing (weighted) mean of the outcome y (whether 0/1
        binary for classification or quantitative for
        regression).

        This function implements this preprocessing step
        and produces a sorted list of category values.

        This function assumes that y is comprised of a single column
        indicating a single outcome target.
        """
        cdef:
            intp_t[:] samples = self.samples
            float32_t[:] feature_values = self.feature_values
            const float64_t[:, ::1] y = self.y
            const float64_t[:] sample_weight = self.sample_weight

            float64_t w
            intp_t cat, localcat
            intp_t q, sample_idx

        # categorical features with more than 64 categories are not supported
        # here.
        self.sort_value[:] = 0
        self.sort_density[:] = 0
        # memset(sort_value, 0, 64 * sizeof(float32_t))
        # memset(sort_density, 0, 64 * sizeof(float32_t))

        for q in range(start, end):
            cat = <intp_t> feature_values[q]
            sample_idx = samples[q]

            if sample_weight is not None:
                w = sample_weight[sample_idx]
            else:
                w = 1.0
            self.sort_value[cat] += w * y[sample_idx, 0]
            self.sort_density[cat] += w

        for localcat in range(ncat_present):
            cat = localcat + cat_offset[localcat]
            if self.sort_density[cat] == 0:  # Avoid dividing by zero
                self.sort_density[cat] = 1
            self.sort_value[localcat] = self.sort_value[cat] / self.sort_density[cat]
            sorted_cat[localcat] = cat

        # cdef inline void sort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil:
        sort(&self.sort_value[0], &sorted_cat[0], ncat_present)


cdef inline bint goes_left(
    float32_t feature_value,
    SplitValue split,
    int32_t n_categories,
    BITSET_t[:] cat_cache
) noexcept nogil:
    """Determine whether a sample goes to the left or right child node.

    For numerical features, ``(-inf, split.threshold]`` is the left child, and
    ``(split.threshold, inf)`` the right child.
    For categorical features, if the corresponding bit for the category is set
    in cachebits, the left child isused, and if not set, the right child. If
    the given input category is larger than the ``n_categories``, the right
    child is assumed.

    Attributes
    ----------
    feature_value : float32_t
        The value of the feature for which the decision needs to be made.
    split : SplitValue
        The union (of float64_t and BITSET_t) indicating the split. However, it
        is used (as a float64_t) only for numerical features.
    n_categories : int32_t
        The number of categories present in the feature in question. The
        feature is considered a numerical one and not a categorical one if
        n_categories is negative.
    cat_cache : BITSET_t*
        The array containing the expansion of split.cat_split. The function
        setup_cat_cache is the one filling it.

    Returns
    -------
    result : bint
        Indicating whether the left branch should be used.
    """
    cdef intp_t idx

    if n_categories < 0:
        # Non-categorical feature
        return feature_value <= split.threshold
    else:
        # Categorical feature, using bit cache
        if (<intp_t> feature_value) < n_categories:
            idx = (<intp_t> feature_value) // 64
            offset = (<intp_t> feature_value) % 64
            return bs_get(cat_cache[idx], offset)
        else:
            return 0


cdef inline void shift_missing_values_to_left_if_required(
    SplitRecord* best,
    intp_t[::1] samples,
    intp_t end,
) noexcept nogil:
    """Shift missing value sample indices to the left of the split if required.

    Note: this should always be called at the very end because it will
    move samples around, thereby affecting the criterion.
    This affects the computation of the children impurity, which affects
    the computation of the next node.
    """
    cdef intp_t i, p, current_end
    # The partitioner partitions the data such that the missing values are in
    # samples[-n_missing:] for the criterion to consume. If the missing values
    # are going to the right node, then the missing values are already in the
    # correct position. If the missing values go left, then we move the missing
    # values to samples[best.pos:best.pos+n_missing] and update `best.pos`.
    if best.n_missing > 0 and best.missing_go_to_left:
        for p in range(best.n_missing):
            i = best.pos + p
            current_end = end - 1 - p
            samples[i], samples[current_end] = samples[current_end], samples[i]
        best.pos += best.n_missing

# Introduce a fused-class to make it possible to share the split implementation
# between the dense and sparse cases in the node_split_best and node_split_random
# functions. The alternative would have been to use inheritance-based polymorphism
# but it would have resulted in a ~10% overall tree fitting performance
# degradation caused by the overhead frequent virtual method lookups.
ctypedef fused Partitioner:
    DensePartitioner
    SparsePartitioner

cdef inline int node_split_best(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """Find the best split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Find the best split
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t end_non_missing
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_searches
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] samples = splitter.samples
    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef float32_t[::1] feature_values = splitter.feature_values
    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = -INFINITY
    cdef float64_t best_proxy_improvement = -INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    # variables for categorical split handling
    cdef bint breiman_shortcut = splitter.breiman_shortcut
    cdef bint is_categorical
    # index through categories
    cdef uint64_t cat_idx
    # total number of categories per feature
    cdef uint64_t ncat_present
    # the bitset to store which category to split on
    cdef BITSET_t cat_split = 0

    # XXX: unsure what this it.
    cdef int32_t[:] cat_offs = splitter.cat_offset

    # A storage of the sorted categories used in Breiman shortcut
    cdef intp_t[:] sorted_cat = splitter.sorted_cat

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    cdef intp_t p
    cdef intp_t p_prev

    cdef intp_t n_visited_features = 0
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants

    cdef intp_t i

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):

        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[
        current_split.feature = features[f_j]
        partitioner.sort_samples_and_feature_values(current_split.feature)
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            # All values for this feature are missing, or
            end_non_missing == start or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
        ):
            # We consider this feature constant in this case.
            # Since finding a split among constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)  # initialize even when n_missing == 0

        is_categorical = splitter.n_categories[current_split.feature] > 0
        if is_categorical:
            # Identify the number of categories present in this node
            # and apply breiman sorting if number of categories is small
            # XXX: could improve this by passing in parent information.
            cat_split = 0
            ncat_present = 0

            # Initialize the bitset for the categories present in the node
            for i in range(start, end):
                # Xf[i] < 64 already verified in tree.py
                cat_split = bs_set(cat_split, <intp_t>feature_values[i])

            # count the number of categories present per feature in this node
            for i in range(splitter.n_categories[current_split.feature]):
                if bs_get(cat_split, i):
                    cat_offs[ncat_present] = i - ncat_present
                    ncat_present += 1

            # TODO: Why do we need to recompute ncat_present? Isn't it in n_categories?
            # - we do it since the number of categories may change as we traverse the tree, but
            # instead of running this loop could we pass in parent information? via parentInfo...
            # similar to constant feature tracking
            if ncat_present <= 3:
                breiman_shortcut = False  # No benefit for small N

            # Apply sorting to the categories if we can leverage the Breiman computational
            # trick to improve the computational efficiency of the categorical splits
            if breiman_shortcut:
                splitter._breiman_sort_categories(
                    start,
                    end,
                    splitter.n_categories[current_split.feature],
                    ncat_present,
                    cat_offs,
                    sorted_cat
                )

        # Evaluate all splits

        # If there are missing values, then we search twice for the most optimal split.
        # The first search will have all the missing values going to the right node.
        # The second search will have all the missing values going to the left node.
        # If there are no missing values, then we search only once for the most
        # optimal split.
        n_searches = 2 if has_missing else 1

        for i in range(n_searches):
            missing_go_to_left = i == 1
            criterion.missing_go_to_left = missing_go_to_left
            criterion.reset()

            p = start
            cat_idx = 0

            while p < end_non_missing:
                if is_categorical:
                    cat_idx += 1

                    if breiman_shortcut:
                        # TODO: Implement breiman shortcut
                        pass
                    else:
                        if cat_idx >= (<uint64_t> 1) << (ncat_present - 1):
                            break

                        # Expand the bits of (2 * cat_idx) out into
                        # cat_split. We double cat_idx to avoid
                        # double-counting equivalent splits. This also
                        # ensures that cat_split & 1 == 0 as required
                        cat_split = bs_from_template(
                            cat_idx << 1,
                            cat_offs, ncat_present)

                    # Partition samples
                    p = partitioner.partition_samples_category(cat_split)

                    # Must reset criterion since we've reordered the samples
                    criterion.reset()
                else:
                    partitioner.next_p(&p_prev, &p)

                    if p >= end_non_missing:
                        continue

                    if missing_go_to_left:
                        n_left = p - start + n_missing
                        n_right = end_non_missing - p
                    else:
                        n_left = p - start
                        n_right = end_non_missing - p + n_missing

                current_split.pos = p

                # Reject if min_samples_leaf is not guaranteed
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                criterion.update(current_split.pos)

                # Reject if monotonicity constraints are not satisfied
                if (
                    with_monotonic_cst and
                    monotonic_cst[current_split.feature] != 0 and
                    not criterion.check_monotonicity(
                        monotonic_cst[current_split.feature],
                        lower_bound,
                        upper_bound,
                    )
                ):
                    continue

                # Reject if min_weight_leaf is not satisfied
                if ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    continue

                current_proxy_improvement = criterion.proxy_impurity_improvement()

                if current_proxy_improvement > best_proxy_improvement:
                    best_proxy_improvement = current_proxy_improvement

                    if is_categorical:
                        current_split.split_value.cat_split = cat_split
                    else:
                        # sum of halves is used to avoid infinite value
                        current_split.split_value.threshold = (
                            feature_values[p_prev] / 2.0 + feature_values[p] / 2.0
                        )

                    if (
                        current_split.split_value.threshold == feature_values[p] or
                        current_split.split_value.threshold == INFINITY or
                        current_split.split_value.threshold == -INFINITY
                    ):
                        current_split.split_value.threshold = feature_values[p_prev]

                    current_split.n_missing = n_missing

                    # if there are no missing values in the training data, during
                    # test time, we send missing values to the branch that contains
                    # the most samples during training time.
                    if n_missing == 0:
                        current_split.missing_go_to_left = n_left > n_right
                    else:
                        current_split.missing_go_to_left = missing_go_to_left

                    best_split = current_split  # copy

        # Evaluate when there are missing values and all missing values goes
        # to the right node and non-missing values goes to the left node.
        if has_missing:
            n_left, n_right = end - start - n_missing, n_missing
            p = end - n_missing
            missing_go_to_left = 0

            if not (n_left < min_samples_leaf or n_right < min_samples_leaf):
                criterion.missing_go_to_left = missing_go_to_left
                criterion.update(p)

                if not ((criterion.weighted_n_left < min_weight_leaf) or
                        (criterion.weighted_n_right < min_weight_leaf)):
                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        current_split.split_value.threshold = INFINITY
                        current_split.missing_go_to_left = missing_go_to_left
                        current_split.n_missing = n_missing
                        current_split.pos = p
                        best_split = current_split

    # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
    if best_split.pos < end:
        setup_cat_cache(
            splitter.cat_cache,
            best_split.split_value.cat_split,
            splitter.n_categories[best_split.feature]
        )

        partitioner.partition_samples_final(
            best_split.pos,
            best_split.split_value,
            best_split.feature,
            best_split.n_missing
        )
        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

        shift_missing_values_to_left_if_required(&best_split, samples, end)

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0


# Sort n-element arrays pointed to by feature_values and samples, simultaneously,
# by the values in feature_values. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil:
    if n == 0:
        return
    cdef intp_t maxd = 2 * <intp_t>log(n)
    introsort(feature_values, samples, n, maxd)


cdef inline void swap(float32_t* feature_values, intp_t* samples,
                      intp_t i, intp_t j) noexcept nogil:
    # Helper for sort
    feature_values[i], feature_values[j] = feature_values[j], feature_values[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline float32_t median3(float32_t* feature_values, intp_t n) noexcept nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef float32_t a = feature_values[0], b = feature_values[n / 2], c = feature_values[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(float32_t* feature_values, intp_t *samples,
                    intp_t n, intp_t maxd) noexcept nogil:
    cdef float32_t pivot
    cdef intp_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(feature_values, samples, n)
            return
        maxd -= 1

        pivot = median3(feature_values, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if feature_values[i] < pivot:
                swap(feature_values, samples, i, l)
                i += 1
                l += 1
            elif feature_values[i] > pivot:
                r -= 1
                swap(feature_values, samples, i, r)
            else:
                i += 1

        introsort(feature_values, samples, l, maxd)
        feature_values += r
        samples += r
        n -= r


cdef inline void sift_down(float32_t* feature_values, intp_t* samples,
                           intp_t start, intp_t end) noexcept nogil:
    # Restore heap order in feature_values[start:end] by moving the max element to start.
    cdef intp_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and feature_values[maxind] < feature_values[child]:
            maxind = child
        if child + 1 < end and feature_values[maxind] < feature_values[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(feature_values, samples, root, maxind)
            root = maxind


cdef void heapsort(float32_t* feature_values, intp_t* samples, intp_t n) noexcept nogil:
    cdef intp_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(feature_values, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(feature_values, samples, 0, end)
        sift_down(feature_values, samples, 0, end)
        end = end - 1

cdef inline int node_split_random(
    Splitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
) except -1 nogil:
    """Find the best random split on node samples[start:end]

    Returns -1 in case of failure to allocate memory (and raise MemoryError)
    or 0 otherwise.
    """
    cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
    cdef bint with_monotonic_cst = splitter.with_monotonic_cst

    # Draw random splits and pick the best
    cdef intp_t start = splitter.start
    cdef intp_t end = splitter.end
    cdef intp_t end_non_missing
    cdef intp_t n_missing = 0
    cdef bint has_missing = 0
    cdef intp_t n_left, n_right
    cdef bint missing_go_to_left

    cdef intp_t[::1] samples = splitter.samples
    cdef intp_t[::1] features = splitter.features
    cdef intp_t[::1] constant_features = splitter.constant_features
    cdef intp_t n_features = splitter.n_features

    cdef intp_t max_features = splitter.max_features
    cdef intp_t min_samples_leaf = splitter.min_samples_leaf
    cdef float64_t min_weight_leaf = splitter.min_weight_leaf
    cdef uint32_t* random_state = &splitter.rand_r_state

    # variables for categorical split handling
    cdef bint is_categorical
    # index through categories
    cdef uint64_t split_seed

    cdef SplitRecord best_split, current_split
    cdef float64_t current_proxy_improvement = - INFINITY
    cdef float64_t best_proxy_improvement = - INFINITY

    cdef float64_t impurity = parent_record.impurity
    cdef float64_t lower_bound = parent_record.lower_bound
    cdef float64_t upper_bound = parent_record.upper_bound

    cdef intp_t f_i = n_features
    cdef intp_t f_j
    # Number of features discovered to be constant during the split search
    cdef intp_t n_found_constants = 0
    # Number of features known to be constant and drawn without replacement
    cdef intp_t n_drawn_constants = 0
    cdef intp_t n_known_constants = parent_record.n_constant_features
    # n_total_constants = n_known_constants + n_found_constants
    cdef intp_t n_total_constants = n_known_constants
    cdef intp_t n_visited_features = 0
    cdef float32_t min_feature_value
    cdef float32_t max_feature_value

    _init_split(&best_split, end)

    partitioner.init_node_split(start, end)

    # Sample up to max_features without replacement using a
    # Fisher-Yates-based algorithm (using the local variables `f_i` and
    # `f_j` to compute a permutation of the `features` array).
    #
    # Skip the CPU intensive evaluation of the impurity criterion for
    # features that were already detected as constant (hence not suitable
    # for good splitting) by ancestor nodes and save the information on
    # newly discovered constant features to spare computation on descendant
    # nodes.
    while (f_i > n_total_constants and  # Stop early if remaining features
                                        # are constant
            (n_visited_features < max_features or
             # At least one drawn features must be non constant
             n_visited_features <= n_found_constants + n_drawn_constants)):
        n_visited_features += 1

        # Loop invariant: elements of features in
        # - [:n_drawn_constant[ holds drawn and known constant features;
        # - [n_drawn_constant:n_known_constant[ holds known constant
        #   features that haven't been drawn yet;
        # - [n_known_constant:n_total_constant[ holds newly found constant
        #   features;
        # - [n_total_constant:f_i[ holds features that haven't been drawn
        #   yet and aren't constant apriori.
        # - [f_i:n_features[ holds features that have been drawn
        #   and aren't constant.

        # Draw a feature at random
        f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                       random_state)

        if f_j < n_known_constants:
            # f_j in the interval [n_drawn_constants, n_known_constants[
            features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]
            n_drawn_constants += 1
            continue

        # f_j in the interval [n_known_constants, f_i - n_found_constants[
        f_j += n_found_constants
        # f_j in the interval [n_total_constants, f_i[

        current_split.feature = features[f_j]

        # Find min, max as we will randomly select a threshold between them
        partitioner.find_min_max(
            current_split.feature, &min_feature_value, &max_feature_value
        )
        n_missing = partitioner.n_missing
        end_non_missing = end - n_missing

        if (
            # All values for this feature are missing, or
            end_non_missing == start or
            # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
            max_feature_value <= min_feature_value + FEATURE_THRESHOLD
        ):
            # We consider this feature constant in this case.
            # Since finding a split with a constant feature is not valuable,
            # we do not consider this feature for splitting.
            features[f_j], features[n_total_constants] = features[n_total_constants], current_split.feature

            n_found_constants += 1
            n_total_constants += 1
            continue

        f_i -= 1
        features[f_i], features[f_j] = features[f_j], features[f_i]
        has_missing = n_missing != 0
        criterion.init_missing(n_missing)

        # Construct a random split
        is_categorical = splitter.n_categories[current_split.feature] > 0
        if is_categorical:
            split_seed = rand_int(0, <uint32_t>RAND_R_MAX + 1, random_state)
            current_split.split_value.cat_split = (split_seed << 32) | 1
        else:
            # Draw a random threshold
            current_split.split_value.threshold = rand_uniform(
                min_feature_value,
                max_feature_value,
                random_state,
            )

            if current_split.split_value.threshold == max_feature_value:
                current_split.split_value.threshold = min_feature_value

        # Partition
        setup_cat_cache(
            splitter.cat_cache,
            current_split.split_value.cat_split,
            splitter.n_categories[current_split.feature]
        )
        current_split.pos = partitioner.partition_samples(
            current_split.split_value,
            current_split.feature
        )

        # Randomly split missing values
        if has_missing:
            # If there are missing values, then we randomly make all missing
            # values go to the right or left.
            #
            # Note: compared to the BestSplitter, we do not evaluate the
            # edge case where all the missing values go to the right node
            # and the non-missing values go to the left node. This is because
            # this would indicate a threshold outside of the observed range
            # of the feature. However, it is not clear how much probability weight should
            # be given to this edge case.
            missing_go_to_left = rand_int(0, 2, random_state)
        else:
            missing_go_to_left = 0
        criterion.missing_go_to_left = missing_go_to_left

        if missing_go_to_left:
            n_left = current_split.pos - start + n_missing
            n_right = end_non_missing - current_split.pos
        else:
            n_left = current_split.pos - start
            n_right = end_non_missing - current_split.pos + n_missing

        # Reject if min_samples_leaf is not guaranteed
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        # Evaluate split
        # At this point, the criterion has a view into the samples that was partitioned
        # by the partitioner. The criterion will use the partition to evaluating the split.
        criterion.reset()
        criterion.update(current_split.pos)

        # Reject if min_weight_leaf is not satisfied
        if ((criterion.weighted_n_left < min_weight_leaf) or
                (criterion.weighted_n_right < min_weight_leaf)):
            continue

        # Reject if monotonicity constraints are not satisfied
        if (
                with_monotonic_cst and
                monotonic_cst[current_split.feature] != 0 and
                not criterion.check_monotonicity(
                    monotonic_cst[current_split.feature],
                    lower_bound,
                    upper_bound,
                )
        ):
            continue

        current_proxy_improvement = criterion.proxy_impurity_improvement()

        if current_proxy_improvement > best_proxy_improvement:
            current_split.n_missing = n_missing

            # if there are no missing values in the training data, during
            # test time, we send missing values to the branch that contains
            # the most samples during training time.
            if has_missing:
                current_split.missing_go_to_left = missing_go_to_left
            else:
                current_split.missing_go_to_left = n_left > n_right

            best_proxy_improvement = current_proxy_improvement
            best_split = current_split  # copy

    # Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if best_split.pos < end:
        setup_cat_cache(
            splitter.cat_cache,
            best_split.split_value.cat_split,
            splitter.n_categories[best_split.feature]
        )

        if current_split.feature != best_split.feature:
            partitioner.partition_samples_final(
                best_split.pos,
                best_split.split_value,
                best_split.feature,
                best_split.n_missing
            )
        criterion.init_missing(best_split.n_missing)
        criterion.missing_go_to_left = best_split.missing_go_to_left

        criterion.reset()
        criterion.update(best_split.pos)
        criterion.children_impurity(
            &best_split.impurity_left, &best_split.impurity_right
        )
        best_split.improvement = criterion.impurity_improvement(
            impurity,
            best_split.impurity_left,
            best_split.impurity_right
        )

        shift_missing_values_to_left_if_required(&best_split, samples, end)

    # Respect invariant for constant features: the original order of
    # element in features[:n_known_constants] must be preserved for sibling
    # and child nodes
    memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

    # Copy newly found constant features
    memcpy(&constant_features[n_known_constants],
           &features[n_known_constants],
           sizeof(intp_t) * n_found_constants)

    # Return values
    parent_record.n_constant_features = n_total_constants
    split[0] = best_split
    return 0


@final
cdef class DensePartitioner:
    """Partitioner specialized for dense data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef:
        const float32_t[:, :] X
        cdef intp_t[::1] samples
        cdef float32_t[::1] feature_values
        cdef intp_t start
        cdef intp_t end
        cdef intp_t n_missing
        cdef const unsigned char[::1] missing_values_in_feature_mask
        cdef const int32_t[::1] n_categories
        cdef BITSET_t[::1] cat_cache

    def __init__(
        self,
        const float32_t[:, :] X,
        intp_t[::1] samples,
        float32_t[::1] feature_values,
        const unsigned char[::1] missing_values_in_feature_mask,
        const int32_t[::1] n_categories,
    ):
        self.X = X
        self.samples = samples
        self.feature_values = feature_values
        self.missing_values_in_feature_mask = missing_values_in_feature_mask

        # Initialize the number of categories for each feature
        # A value of -1 indicates a non-categorical feature
        n_features = X.shape[1]
        if n_categories is None:
            self.n_categories = np.array([-1] * n_features, dtype=np.int32)
        else:
            self.n_categories = np.empty(n_categories, dtype=np.int32)
            self.n_categories[:] = n_categories

        # If needed, allocate cache space for categorical splits
        cdef int32_t max_n_categories = max(self.n_categories)
        if max_n_categories > 0:
            cache_size = (max_n_categories + 63) // 64
            self.cat_cache[:] = np.empty(cache_size, dtype=np.uint32)
        else:
            self.cat_cache[:] = np.empty(1, dtype=np.uint32)

    cdef inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil:
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.n_missing = 0

    cdef inline void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil:
        """Simultaneously sort based on the feature_values.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """
        cdef:
            intp_t i, current_end
            float32_t[::1] feature_values = self.feature_values
            const float32_t[:, :] X = self.X
            intp_t[::1] samples = self.samples
            intp_t n_missing = 0
            const unsigned char[::1] missing_values_in_feature_mask = self.missing_values_in_feature_mask

        # Sort samples along that feature; by
        # copying the values into an array and
        # sorting the array in a manner which utilizes the cache more
        # effectively.
        if missing_values_in_feature_mask is not None and missing_values_in_feature_mask[current_feature]:
            i, current_end = self.start, self.end - 1
            # Missing values are placed at the end and do not participate in the sorting.
            while i <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values at its left.
                if isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if isnan(X[samples[i], current_feature]):
                    samples[i], samples[current_end] = samples[current_end], samples[i]
                    n_missing += 1
                    current_end -= 1

                feature_values[i] = X[samples[i], current_feature]
                i += 1
        else:
            # When there are no missing values, we only need to copy the data into
            # feature_values
            for i in range(self.start, self.end):
                feature_values[i] = X[samples[i], current_feature]

        sort(&feature_values[self.start], &samples[self.start], self.end - self.start - n_missing)
        self.n_missing = n_missing

    cdef inline void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil:
        """Find the minimum and maximum value for current_feature.

        Missing values are stored at the end of feature_values.
        The number of missing values observed in feature_values is stored
        in self.n_missing.
        """
        cdef:
            intp_t p, current_end
            float32_t current_feature_value
            const float32_t[:, :] X = self.X
            intp_t[::1] samples = self.samples
            float32_t min_feature_value = INFINITY_32t
            float32_t max_feature_value = -INFINITY_32t
            float32_t[::1] feature_values = self.feature_values
            intp_t n_missing = 0
            const unsigned char[::1] missing_values_in_feature_mask = self.missing_values_in_feature_mask

        # We are copying the values into an array and
        # finding min/max of the array in a manner which utilizes the cache more
        # effectively. We need to also count the number of missing-values there are
        if missing_values_in_feature_mask is not None and missing_values_in_feature_mask[current_feature]:
            p, current_end = self.start, self.end - 1
            # Missing values are placed at the end and do not participate in the
            # min/max calculation.
            while p <= current_end:
                # Finds the right-most value that is not missing so that
                # it can be swapped with missing values towards its left.
                if isnan(X[samples[current_end], current_feature]):
                    n_missing += 1
                    current_end -= 1
                    continue

                # X[samples[current_end], current_feature] is a non-missing value
                if isnan(X[samples[p], current_feature]):
                    samples[p], samples[current_end] = samples[current_end], samples[p]
                    n_missing += 1
                    current_end -= 1

                current_feature_value = X[samples[p], current_feature]
                feature_values[p] = current_feature_value
                if current_feature_value < min_feature_value:
                    min_feature_value = current_feature_value
                elif current_feature_value > max_feature_value:
                    max_feature_value = current_feature_value
                p += 1
        else:
            min_feature_value = X[samples[self.start], current_feature]
            max_feature_value = min_feature_value

            feature_values[self.start] = min_feature_value
            for p in range(self.start + 1, self.end):
                current_feature_value = X[samples[p], current_feature]
                feature_values[p] = current_feature_value

                if current_feature_value < min_feature_value:
                    min_feature_value = current_feature_value
                elif current_feature_value > max_feature_value:
                    max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value
        max_feature_value_out[0] = max_feature_value
        self.n_missing = n_missing

    cdef inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil:
        """Compute the next p_prev and p for iteratiing over feature values.

        The missing values are not included when iterating through the feature values.
        """
        cdef:
            float32_t[::1] feature_values = self.feature_values
            intp_t end_non_missing = self.end - self.n_missing

        while (
            p[0] + 1 < end_non_missing and
            feature_values[p[0] + 1] <= feature_values[p[0]] + FEATURE_THRESHOLD
        ):
            p[0] += 1

        p_prev[0] = p[0]

        # By adding 1, we have
        # (feature_values[p] >= end) or (feature_values[p] > feature_values[p - 1])
        p[0] += 1

    cdef inline intp_t partition_samples(
        self,
        # float64_t current_threshold
        SplitValue split_value,
        intp_t feature,
    ) noexcept nogil:
        """Partition samples for feature_values at the current_threshold."""
        cdef:
            intp_t p = self.start
            intp_t partition_end = self.end
            intp_t[::1] samples = self.samples
            float32_t[::1] feature_values = self.feature_values
            const int32_t[:] n_categories = self.n_categories
            BITSET_t[:] cat_cache = self.cat_cache

        while p < partition_end:
            # if feature_values[p] <= current_threshold:
            if goes_left(
                feature_values[p],
                split_value,
                n_categories[feature],
                cat_cache,
            ):
                p += 1
            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )
                samples[p], samples[partition_end] = samples[partition_end], samples[p]

        return partition_end

    cdef inline intp_t partition_samples_category(self, BITSET_t cat_split) noexcept nogil:
        cdef:
            intp_t p = self.start
            intp_t partition_end = self.end
            intp_t[::1] samples = self.samples
            float32_t[::1] feature_values = self.feature_values

        while p < partition_end:
            # XXX: is casting necessary?
            if bs_get(cat_split, <intp_t>feature_values[p]):
                p += 1
            else:
                partition_end -= 1
                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p])
                samples[p], samples[partition_end] = (
                    samples[partition_end], samples[p])
        return partition_end

    cdef inline void partition_samples_final(
        self,
        intp_t best_pos,
        # float64_t best_threshold,
        SplitValue best_split_value,
        intp_t best_feature,
        intp_t best_n_missing,
    ) noexcept nogil:
        """Partition samples for X at the best_threshold and best_feature.

        If missing values are present, this method partitions `samples`
        so that the `best_n_missing` missing values' indices are in the
        right-most end of `samples`, that is `samples[end_non_missing:end]`.
        """
        cdef:
            # Local invariance: start <= p <= partition_end <= end
            intp_t start = self.start
            intp_t p = start
            intp_t end = self.end - 1
            intp_t partition_end = end - best_n_missing
            intp_t[::1] samples = self.samples
            const float32_t[:, :] X = self.X
            float32_t current_value
            const int32_t[:] n_categories = self.n_categories
            BITSET_t[:] cat_cache = self.cat_cache

        if best_n_missing != 0:
            # Move samples with missing values to the end while partitioning the
            # non-missing samples
            while p < partition_end:
                # Keep samples with missing values at the end
                if isnan(X[samples[end], best_feature]):
                    end -= 1
                    continue

                # Swap sample with missing values with the sample at the end
                current_value = X[samples[p], best_feature]
                if isnan(current_value):
                    samples[p], samples[end] = samples[end], samples[p]
                    end -= 1

                    # The swapped sample at the end is always a non-missing value, so
                    # we can continue the algorithm without checking for missingness.
                    current_value = X[samples[p], best_feature]

                # Partition the non-missing samples
                # if current_value <= best_threshold:
                if goes_left(
                    current_value,
                    best_split_value,
                    n_categories[best_feature],
                    cat_cache,
                ):
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1
        else:
            # Partitioning routine when there are no missing values
            while p < partition_end:
                # if X[samples[p], best_feature] <= best_threshold:
                if goes_left(
                    X[samples[p], best_feature],
                    best_split_value,
                    n_categories[best_feature],
                    cat_cache,
                ):
                    p += 1
                else:
                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
                    partition_end -= 1


@final
cdef class SparsePartitioner:
    """Partitioner specialized for sparse CSC data.

    Note that this partitioner is agnostic to the splitting strategy (best vs. random).
    """
    cdef intp_t[::1] samples
    cdef float32_t[::1] feature_values
    cdef intp_t start
    cdef intp_t end
    cdef intp_t n_missing
    cdef const unsigned char[::1] missing_values_in_feature_mask
    cdef const int32_t[::1] n_categories
    cdef BITSET_t[::1] cat_cache

    cdef const float32_t[::1] X_data
    cdef const int32_t[::1] X_indices
    cdef const int32_t[::1] X_indptr

    cdef intp_t n_total_samples

    cdef intp_t[::1] index_to_samples
    cdef intp_t[::1] sorted_samples

    cdef intp_t start_positive
    cdef intp_t end_negative
    cdef bint is_samples_sorted

    def __init__(
        self,
        object X,
        intp_t[::1] samples,
        intp_t n_samples,
        float32_t[::1] feature_values,
        const unsigned char[::1] missing_values_in_feature_mask,
        const int32_t[::1] n_categories
    ):
        if not (issparse(X) and X.format == "csc"):
            raise ValueError("X should be in csc format")

        self.samples = samples
        self.feature_values = feature_values

        # Initialize X
        cdef intp_t n_total_samples = X.shape[0]

        self.X_data = X.data
        self.X_indices = X.indices
        self.X_indptr = X.indptr
        self.n_total_samples = n_total_samples

        # Initialize auxiliary array used to perform split
        self.index_to_samples = np.full(n_total_samples, fill_value=-1, dtype=np.intp)
        self.sorted_samples = np.empty(n_samples, dtype=np.intp)

        cdef intp_t p
        for p in range(n_samples):
            self.index_to_samples[samples[p]] = p

        self.missing_values_in_feature_mask = missing_values_in_feature_mask

        # Initialize the number of categories for each feature
        # A value of -1 indicates a non-categorical feature
        n_features = X.shape[1]
        if n_categories is None:
            self.n_categories = np.array([-1] * n_features, dtype=np.int32)
        else:
            self.n_categories = np.empty(n_categories, dtype=np.int32)
            self.n_categories[:] = n_categories

        # If needed, allocate cache space for categorical splits
        cdef int32_t max_n_categories = max(self.n_categories)
        if max_n_categories > 0:
            cache_size = (max_n_categories + 63) // 64
            self.cat_cache[:] = np.empty(cache_size, dtype=np.uint32)
        else:
            self.cat_cache[:] = np.empty(1, dtype=np.uint32)

    cdef inline void init_node_split(self, intp_t start, intp_t end) noexcept nogil:
        """Initialize splitter at the beginning of node_split."""
        self.start = start
        self.end = end
        self.is_samples_sorted = 0
        self.n_missing = 0

    cdef inline void sort_samples_and_feature_values(
        self, intp_t current_feature
    ) noexcept nogil:
        """Simultaneously sort based on the feature_values."""
        cdef:
            float32_t[::1] feature_values = self.feature_values
            intp_t[::1] index_to_samples = self.index_to_samples
            intp_t[::1] samples = self.samples

        self.extract_nnz(current_feature)
        # Sort the positive and negative parts of `feature_values`
        sort(&feature_values[self.start], &samples[self.start], self.end_negative - self.start)
        if self.start_positive < self.end:
            sort(
                &feature_values[self.start_positive],
                &samples[self.start_positive],
                self.end - self.start_positive
            )

        # Update index_to_samples to take into account the sort
        for p in range(self.start, self.end_negative):
            index_to_samples[samples[p]] = p
        for p in range(self.start_positive, self.end):
            index_to_samples[samples[p]] = p

        # Add one or two zeros in feature_values, if there is any
        if self.end_negative < self.start_positive:
            self.start_positive -= 1
            feature_values[self.start_positive] = 0.

            if self.end_negative != self.start_positive:
                feature_values[self.end_negative] = 0.
                self.end_negative += 1

        # XXX: When sparse supports missing values, this should be set to the
        # number of missing values for current_feature
        self.n_missing = 0

    cdef inline void find_min_max(
        self,
        intp_t current_feature,
        float32_t* min_feature_value_out,
        float32_t* max_feature_value_out,
    ) noexcept nogil:
        """Find the minimum and maximum value for current_feature."""
        cdef:
            intp_t p
            float32_t current_feature_value, min_feature_value, max_feature_value
            float32_t[::1] feature_values = self.feature_values

        self.extract_nnz(current_feature)

        if self.end_negative != self.start_positive:
            # There is a zero
            min_feature_value = 0
            max_feature_value = 0
        else:
            min_feature_value = feature_values[self.start]
            max_feature_value = min_feature_value

        # Find min, max in feature_values[start:end_negative]
        for p in range(self.start, self.end_negative):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        # Update min, max given feature_values[start_positive:end]
        for p in range(self.start_positive, self.end):
            current_feature_value = feature_values[p]

            if current_feature_value < min_feature_value:
                min_feature_value = current_feature_value
            elif current_feature_value > max_feature_value:
                max_feature_value = current_feature_value

        min_feature_value_out[0] = min_feature_value
        max_feature_value_out[0] = max_feature_value

    cdef inline void next_p(self, intp_t* p_prev, intp_t* p) noexcept nogil:
        """Compute the next p_prev and p for iteratiing over feature values."""
        cdef:
            intp_t p_next
            float32_t[::1] feature_values = self.feature_values

        if p[0] + 1 != self.end_negative:
            p_next = p[0] + 1
        else:
            p_next = self.start_positive

        while (p_next < self.end and
                feature_values[p_next] <= feature_values[p[0]] + FEATURE_THRESHOLD):
            p[0] = p_next
            if p[0] + 1 != self.end_negative:
                p_next = p[0] + 1
            else:
                p_next = self.start_positive

        p_prev[0] = p[0]
        p[0] = p_next

    cdef inline intp_t partition_samples(
        self,
        # float64_t current_threshold
        SplitValue split_value,
        intp_t feature,
    ) noexcept nogil:
        """Partition samples for feature_values at the current_threshold."""
        return self._partition(split_value.threshold, self.start_positive)

    cdef inline void partition_samples_final(
        self,
        intp_t best_pos,
        SplitValue best_split_value,
        # float64_t best_threshold,
        intp_t best_feature,
        intp_t n_missing,
    ) noexcept nogil:
        """Partition samples for X at the best_threshold and best_feature."""
        self.extract_nnz(best_feature)
        self._partition(best_split_value.threshold, best_pos)

    cdef inline intp_t _partition(self, float64_t threshold, intp_t zero_pos) noexcept nogil:
        """Partition samples[start:end] based on threshold."""
        # TODO: implement partitioning samples based on categorical split
        cdef:
            intp_t p, partition_end
            intp_t[::1] index_to_samples = self.index_to_samples
            float32_t[::1] feature_values = self.feature_values
            intp_t[::1] samples = self.samples

        if threshold < 0.:
            p = self.start
            partition_end = self.end_negative
        elif threshold > 0.:
            p = self.start_positive
            partition_end = self.end
        else:
            # Data are already split
            return zero_pos

        while p < partition_end:
            if feature_values[p] <= threshold:
                p += 1

            else:
                partition_end -= 1

                feature_values[p], feature_values[partition_end] = (
                    feature_values[partition_end], feature_values[p]
                )
                sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    cdef inline intp_t partition_samples_category(self, BITSET_t cat_split) noexcept nogil:
        cdef:
            intp_t p = self.start
            # intp_t partition_end = self.end
            # intp_t[::1] samples = self.samples
            # float32_t[::1] feature_values = self.feature_values
        # TODO: implement partition samples category for sparse input
        # Right now, we return a dummy value to make compilation work
        return p

    cdef inline void extract_nnz(self, intp_t feature) noexcept nogil:
        """Extract and partition values for a given feature.

        The extracted values are partitioned between negative values
        feature_values[start:end_negative[0]] and positive values
        feature_values[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : intp_t,
            Index of the feature we want to extract non zero value.
        """
        cdef intp_t[::1] samples = self.samples
        cdef float32_t[::1] feature_values = self.feature_values
        cdef intp_t indptr_start = self.X_indptr[feature],
        cdef intp_t indptr_end = self.X_indptr[feature + 1]
        cdef intp_t n_indices = <intp_t>(indptr_end - indptr_start)
        cdef intp_t n_samples = self.end - self.start
        cdef intp_t[::1] index_to_samples = self.index_to_samples
        cdef intp_t[::1] sorted_samples = self.sorted_samples
        cdef const int32_t[::1] X_indices = self.X_indices
        cdef const float32_t[::1] X_data = self.X_data

        # Use binary search if n_samples * log(n_indices) <
        # n_indices and index_to_samples approach otherwise.
        # O(n_samples * log(n_indices)) is the running time of binary
        # search and O(n_indices) is the running time of index_to_samples
        # approach.
        if ((1 - self.is_samples_sorted) * n_samples * log(n_samples) +
                n_samples * log(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
            extract_nnz_binary_search(X_indices, X_data,
                                      indptr_start, indptr_end,
                                      samples, self.start, self.end,
                                      index_to_samples,
                                      feature_values,
                                      &self.end_negative, &self.start_positive,
                                      sorted_samples, &self.is_samples_sorted)

        # Using an index to samples  technique to extract non zero values
        # index_to_samples is a mapping from X_indices to samples
        else:
            extract_nnz_index_to_samples(X_indices, X_data,
                                         indptr_start, indptr_end,
                                         samples, self.start, self.end,
                                         index_to_samples,
                                         feature_values,
                                         &self.end_negative, &self.start_positive)


cdef int compare_intp_t(const void* a, const void* b) noexcept nogil:
    """Comparison function for sort.

    This must return an `int` as it is used by stdlib's qsort, which expects
    an `int` return value.
    """
    return <int>((<intp_t*>a)[0] - (<intp_t*>b)[0])


cdef inline void binary_search(const int32_t[::1] sorted_array,
                               int32_t start, int32_t end,
                               intp_t value, intp_t* index,
                               int32_t* new_start) noexcept nogil:
    """Return the index of value in the sorted array.

    If not found, return -1. new_start is the last pivot + 1
    """
    cdef int32_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(const int32_t[::1] X_indices,
                                              const float32_t[::1] X_data,
                                              int32_t indptr_start,
                                              int32_t indptr_end,
                                              intp_t[::1] samples,
                                              intp_t start,
                                              intp_t end,
                                              intp_t[::1] index_to_samples,
                                              float32_t[::1] feature_values,
                                              intp_t* end_negative,
                                              intp_t* start_positive) noexcept nogil:
    """Extract and partition values for a feature using index_to_samples.

    Complexity is O(indptr_end - indptr_start).
    """
    cdef int32_t k
    cdef intp_t index
    cdef intp_t end_negative_ = start
    cdef intp_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void extract_nnz_binary_search(const int32_t[::1] X_indices,
                                           const float32_t[::1] X_data,
                                           int32_t indptr_start,
                                           int32_t indptr_end,
                                           intp_t[::1] samples,
                                           intp_t start,
                                           intp_t end,
                                           intp_t[::1] index_to_samples,
                                           float32_t[::1] feature_values,
                                           intp_t* end_negative,
                                           intp_t* start_positive,
                                           intp_t[::1] sorted_samples,
                                           bint* is_samples_sorted) noexcept nogil:
    """Extract and partition values for a given feature using binary search.

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    cdef intp_t n_samples

    if not is_samples_sorted[0]:
        n_samples = end - start
        memcpy(&sorted_samples[start], &samples[start],
               n_samples * sizeof(intp_t))
        qsort(&sorted_samples[start], n_samples, sizeof(intp_t),
              compare_intp_t)
        is_samples_sorted[0] = 1

    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    cdef intp_t p = start
    cdef intp_t index
    cdef intp_t k
    cdef intp_t end_negative_ = start
    cdef intp_t start_positive_ = end

    while (p < end and indptr_start < indptr_end):
        # Find index of sorted_samples[p] in X_indices
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
            # If k != -1, we have found a non zero value

            if X_data[k] > 0:
                start_positive_ -= 1
                feature_values[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)

            elif X_data[k] < 0:
                feature_values[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(intp_t[::1] index_to_samples, intp_t[::1] samples,
                             intp_t pos_1, intp_t pos_2) noexcept nogil:
    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
    samples[pos_1], samples[pos_2] = samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2


cdef class BestSplitter(Splitter):
    """Splitter for finding the best split on dense data.

    breiman_shortcut : bint
        Whether we use the Breiman shortcut method when splitting
        a categorical feature.
    """
    cdef DensePartitioner partitioner

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
        const int32_t[::1] n_categories,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask, n_categories)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask, n_categories
        )

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class BestSparseSplitter(Splitter):
    """Splitter for finding the best split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
        const int32_t[::1] n_categories,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask, n_categories)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask, n_categories
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSplitter(Splitter):
    """Splitter for finding the best random split on dense data."""
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
        const int32_t[::1] n_categories,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask, n_categories)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask, n_categories
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )

cdef class RandomSparseSplitter(Splitter):
    """Splitter for finding the best random split, using the sparse data."""
    cdef SparsePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const unsigned char[::1] missing_values_in_feature_mask,
        const int32_t[::1] n_categories,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask, n_categories)
        self.partitioner = SparsePartitioner(
            X, self.samples, self.n_samples, self.feature_values, missing_values_in_feature_mask, n_categories
        )
    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_random(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )
