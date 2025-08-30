#ifndef XSTUITE_GETBIT_H
#define XSTUITE_GETBIT_H

#define GET_BIT(flags, flag_index) \
    (((flags) >> (flag_index)) & 1ULL)

#endif // XSTUITE_GETBIT_H