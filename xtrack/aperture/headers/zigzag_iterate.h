#ifndef XTRACK_ZIGZAG_ITERATE_H
#define XTRACK_ZIGZAG_ITERATE_H

typedef struct {
    int32_t index;
    int32_t offset;
    int32_t start;
    int32_t upper_bound;
    int32_t visited;
    uint8_t wrap;
} ZigZagIterator;


ZigZagIterator zigzag_iterator_new(uint32_t start, uint32_t upper_bound, uint8_t wrap)
/*
    Return a new zigzag iterator: already pointing to `start`, which must be in `[0, upper_bound)`.

    The iterator's `.index` property points to the current index, and iterates within `[0, upper_bound)`,
    either wrapping around the edges of the interval or stopping at the boundaries, according to `wrap`.
*/
{
    return (ZigZagIterator) {
        .index = start,
        .offset = 0,
        .start = start,
        .upper_bound = upper_bound,
        .wrap = wrap,
    };
}


static inline uint8_t zigzag_iterator_next_wrapping(ZigZagIterator* iter)
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
        if (2 * iter->offset == span) return 0;
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


static inline uint8_t zigzag_iterator_next_bounded(ZigZagIterator* iter)
/*
    Advance the zigzag iterator to the next index.

    The iterator alternates outward: start, start+1, start-1, start+2, start-2, ...
    When one side reaches a bound, iteration continues only on the remaining side
    until all indices in [0, upper_bound) are exhausted.

    Returns
    -------
      1 if the iterator was successfully advanced and `iter->index` is valid.
      0 if the iterator is exhausted, and cannot be advanced anymore.
 */
{
    int32_t prev_index = iter->index;

    if (iter->offset == 0) {
        /* Initial condition */
        iter->offset++;
        iter->index++;
    }
    else if (iter->offset > 0) {
        /* Positive side -> negative side */
        iter->index -= 2 * iter->offset;

        if (iter->index < 0) {
            /* Hit the lower bound, continue the positive side */
            iter->index = prev_index + 1;
            iter->offset++;
        }
        else iter->offset *= -1;
    }
    else if (iter->offset < 0) {
        /* Negative side -> positive side + 1 */
        iter->index += -2 * iter->offset + 1;
        if (iter->index >= iter->upper_bound) {
            /* Hit the upper bound, continue the negative side */
            iter->index = prev_index - 1;
            iter->offset--;
        }
        else iter->offset = 1 - iter->offset;
    }

    /* If still not in the bounds, means we've exhausted the iterator */
    if (iter->index < 0 || iter->upper_bound <= iter->index) return 0;
    else return 1;
}


uint8_t zigzag_iterator_next(ZigZagIterator* iter)
{
    if (iter->wrap) return zigzag_iterator_next_wrapping(iter);
    else return zigzag_iterator_next_bounded(iter);
}

#endif /* XTRACK_ZIGZAG_ITERATE_H */
