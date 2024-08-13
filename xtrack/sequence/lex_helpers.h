// A common macro to keep track of token locations in `yylloc`
#define YY_USER_ACTION                                                 \
    yylloc->first_line = yylloc->last_line;                            \
    yylloc->first_column = yylloc->last_column;                        \
    for (int i = 0; yytext[i] != '\0'; i++) {                          \
        if (yytext[i] == '\n') {                                       \
            yylloc->last_line++;                                       \
            yylloc->last_column = 1;                                   \
        } else {                                                       \
            yylloc->last_column++;                                     \
        }                                                              \
    }

// Parse the current token values (in `yytext`) as numeric values
double parse_float(char* text, char* error) {
    char* end_ptr;
    double value = strtod(text, &end_ptr);
    if (end_ptr == text) {
        *error = 1;
    }
    return value;
}

long parse_integer(char* text, char* error) {
    char* end_ptr;
    long value = strtol(text, &end_ptr, 10);
    if (end_ptr == text) {
        *error = 1;
    }
    return value;
}