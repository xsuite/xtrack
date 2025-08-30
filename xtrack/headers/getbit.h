#ifndef XSTUITE_GET_FLAGS_H
#define XSTUITE_GET_FLAGS_H

#define GET_BIT(flags, flag_index) \
    (((flags) >> (flag_index)) & 1ULL)

#endif // XSTUITE_GET_FLAGS_H