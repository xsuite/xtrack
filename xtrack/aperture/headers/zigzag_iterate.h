#ifndef XTRACK_ZIGZAG_ITERATE_H
#define XTRACK_ZIGZAG_ITERATE_H

typedef struct {
    int32_t index;
    int32_t offset;
    int32_t start;
    int32_t lower_bound;
    int32_t upper_bound;
} ZigZagIterator;


ZigZagIterator zigzag_iterator_new(uint32_t start, uint32_t lower_bound, uint32_t upper_bound)
/*
    Return a new zigzag iterator: already pointing to `start`.

    The iterator's `.index` property points to the current index, and iterates within `[lower_bound, upper_bound)`.
*/
{
    return (ZigZagIterator) {
        .index = start,
        .offset = 0,
        .start = start,
        .lower_bound = lower_bound,
        .upper_bound = upper_bound
    };
}


uint8_t zigzag_iterator_next(ZigZagIterator* iter)
/*
    Advance the zigzag iterator to the next index.

    The iterator alternates outward: start, start+1, start-1, start+2, start-2, ...
    When one side reaches a bound, iteration continues only on the remaining side
    until all indices in [lower_bound, upper_bound) are exhausted.

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

        if (iter->index < iter->lower_bound) {
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
    if (iter->index < iter->lower_bound || iter->upper_bound <= iter->index) return 0;
    else return 1;
}

#endif /* XTRACK_ZIGZAG_ITERATE_H */
