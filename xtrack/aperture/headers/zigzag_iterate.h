#ifndef XTRACK_ZIGZAG_ITERATE_H
#define XTRACK_ZIGZAG_ITERATE_H

typedef struct {
    int32_t index;
    int32_t offset;
    int32_t start;
    int32_t upper_bound;
} ZigZagIterator;


ZigZagIterator zigzag_iterator_new(uint32_t start, uint32_t upper_bound)
/*
    Return a new zigzag iterator: already pointing to `start`, which must be in `[0, upper_bound)`.

    The iterator's `.index` property points to the current index, and iterates within
    `[0, upper_bound)` wrapping around the edges of the interval.
*/
{
    return (ZigZagIterator) {
        .index = start,
        .offset = 0,
        .start = start,
        .upper_bound = upper_bound
    };
}


uint8_t zigzag_iterator_next(ZigZagIterator* iter)
/*
    Advance the zigzag iterator to the next index.

    The iterator alternates outward on the circular interval:
    start, start+1, start-1, start+2, start-2, ... modulo upper_bound

    Returns
    -------
      1 if the iterator was successfully advanced and `iter->index` is valid.
      0 if the iterator is exhausted, and cannot be advanced anymore.

    Notes
    -----
    It can be verified that if `upper_bound` is odd, the last visited index will be on a negative `offset`,
    and if `upper_bound` is even, the last visited index will be on a positive `offset`. To determine when
    to terminate the iteration, we need to check if the new offset will cause us to re-visit an index.

    For the even `upper_bound` case, last allowed offset is simply `upper_bound / 2`, and so in the + -> -
    transition, we check for this condition, and if it's met, we terminate.

    For the odd `upper_bound` case, the last allowed offset is `-(upper_bound - 1) / 2 == -floor(upper_bound / 2)`.
    In such a case, the next offset (disallowed) would be `floor(upper_bound) / 2 + 1`, so we check if
    `next_offset > floor(upper_bound) / 2` and terminate if the condition is met.
 */
{
    const int32_t span = iter->upper_bound;
    if (span <= 1) return 0;

    int32_t next_offset;
    if (iter->offset == 0) {
        /* Initial condition */
        next_offset = 1;
    }
    else if (iter->offset > 0) {
        /* Positive side -> negative side */
        if (iter->offset == span / 2) return 0;
        next_offset = -iter->offset;
    }
    else {
        /* Negative side -> positive side */
        next_offset = -iter->offset + 1;
        if (next_offset > span / 2) return 0;
    }

    /* Compute the index */
    int32_t rel_index = iter->start + next_offset;
    rel_index %= span;
    if (rel_index < 0) rel_index += span;

    iter->offset = next_offset;
    iter->index = rel_index;
    return 1;
}

#endif /* XTRACK_ZIGZAG_ITERATE_H */
