#cython: language_level=3
# This file defines the interface between the C part of the parser and the
# Python/Cython modules. We expose the C constructs we need from flex/bison code
# to Python, and declare the C functions ('yyerror') that are expected by the
# parser, which we then implement in the corresponding Cython 'xmad.py' file.

from libc.stdio cimport FILE

cdef extern from "xmad_lex.h":
    struct yy_buffer_state:
        pass

    yy_buffer_state* yy_scan_string(const char *)
    int yylineno
    FILE* yyin
    const char* yytext

cdef extern from "xmad_tab.h":
    struct YYLTYPE:
        int first_line
        int first_column
        int last_line
        int last_column

    YYLTYPE yylloc

cdef extern int yyparse()
cdef extern int yylex_destroy()

cdef public void yyerror(const char* message)

cdef public object py_float(double value)
cdef public object py_unary_op(const char* op_string, object value)
cdef public object py_binary_op(const char* op_string, object left, object right)
cdef public object py_eq_value_scalar(const char* identifier, object value)
cdef public object py_eq_defer_scalar(const char* identifier, object value)
cdef public object py_call_func(const char* func_name, object value)