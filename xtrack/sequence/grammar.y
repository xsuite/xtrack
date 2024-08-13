/* copyright *********************************
 * This file is part of the Xtrack Package.  *
 * Copyright (c) CERN, 2024.                 *
 *********************************************

 A parser definition for a MAD-X inspired lattice description language.
 The grammar is built on top of the tokens defined in this file, and produced
 by the lexer, whose definition can be found in the accompanying *.l file.
 The parser interfaces with Python through Cython bindings that are declared
 in the *.pxd file and implemented in the *.py file. The convention of this
 file is that these functions start with a `py_` prefix.

 The grammar
 -----------
 Use left-recursion to minimise stack pressure throughout the grammar. Anyways,
 this makes the rules nicer for lists, as we can simply append.

 Some notes on error handling
 ----------------------------
 The `py_*` functions below can in principle produce python exceptions (they
 execute python code). When that happens, the function returns a null pointer,
 which in our C code will quickly lead to a SIGSEGV. To avoid that, make sure
 to catch all errors before we let them bubble up to the C code. More info:
 https://docs.python.org/3/extending/extending.html#intermezzo-errors-and-exceptions

 The special `error` rule can be used to handle errors in a non-fatal way in
 the grammar. It is a special token that is matched when other rules fail,
 and can be used strategically to recover from some problems in a way that
 produces sensible errors to the user.

 C-side memory management
 ------------------------
 String token values are `strdup`ed into `yylval`, as it is unsafe to share the
 `yytext` pointer outside the *.l file, as its lifetime is generally only
 guaranteed within a lexer rule. This means, however, that in order to avoid
 leaking memory every value of `yylval` needs to be freed after use. Let's
 adopt the following convention within this parser: each `py_` function that
 takes a C-string is responsible for an immediate conversion of the string into
 PyUnicode (or any kind of memorisation, copy, etc., as needed), so that at the
 end of every grammar rule action we can safely free the values.

 In any case it is for the best to avoid passing C pointers on the Python side
 whenever possible, so let us take care of them as close to the C-Python
 boundary as we can.
*/
// Enable storing line & column no with yylloc (see YY_USER_ACTION macro in *.l)
%locations
// Enable re-entrancy (thread safety!)
%define api.pure
// Allegedly provides better error messages
%define parse.error verbose
// As-is the grammar needs 2 lookahead token to distinguish between line_body
// and line_commands (see shift/reduce conflict warning). We enable GLR parsing
// to work around this problem, and seeing that this only happens once per
// beamline, and it is only 2 lookahead, the performance hit will be negligible.
%glr-parser

%{
#define YYERROR_VERBOSE

#include <stdlib.h>
#include <stdio.h>
#include <Python.h>

#include "parser_tab.h"
#include "lexer.h"
#include "parser.h"
%}

%lex-param {void* yyscanner}
%parse-param {void* yyscanner}

// The union type that store token values
%union {
    double floating;
    long integer;
    char* string;
    PyObject* object;
}

// Basic syntax
%token PAREN_OPEN		"("
%token PAREN_CLOSE		")"
%token BRACE_OPEN		"{"
%token BRACE_CLOSE		"}"
%token COLON			":"
%token COMMA			","
%token SEMICOLON		";"
// Keywords
%token STARTLINE		"beamline"
%token ENDLINE			"endbeamline"
%token TRUE			"true"
%token FALSE			"false"
// Values
%token<floating> FLOAT		"floating point number"
%token<integer> INTEGER		"integer number"
%token<string> IDENTIFIER	"identifier"
%token<string> STRING_LITERAL	"string literal"
// Assignments
%token ASSIGN			"="
// Comparisons
%token EQ			"=="
%token NE			"!="
%token GT			">"
%token GE			">="
%token LT			"<"
%token LE			"<="
// Operators
// TODO: Logical operators are supported by xdeps (can implement a ternary op.)
%token ADD			"+"
%token SUB			"-"
%token MUL			"*"
%token DIV			"/"
%token MOD			"%"
%token POW			"^"
// Accessor
%token ARROW			"->"

// Nonterminal (rule) types
%type <object> atom power product sum reference boolean
%type <object> clone argument command
%type <object> argument_assign flag variable_assign
%type <object> arguments array scalar_list
%type <object> line_head line_body line_commands

// Clean up token values on error
%destructor { free($$); } <string>
// Clean up the python objects properly on error
%destructor { Py_DECREF($$); } <object>

// Associativity rules
%left ADD SUB
%left MUL DIV MOD
%left EQ NE GT GE LT LE

%%

start
	: statement
	| start statement

statement
	: set_value
	| clone				{
			py_clone_global(yyscanner, $1);
			Py_XDECREF($1);
		}
	| line
	| error ";"  // Recover from an erroneous line.

set_value
	: variable_assign ";"	{
			py_set_value(yyscanner, $1, @1);
			Py_XDECREF($1);
		}
	| reference "=" sum ";"	{
			py_set_ref(yyscanner, $1, $3);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}
	| reference "=" boolean ";"	{
			py_set_ref(yyscanner, $1, $3);
			Py_XDECREF($1);
		}

variable_assign
	: IDENTIFIER "=" sum		{
			$$ = py_assign(yyscanner, $1, $3);
			free($1);
			Py_XDECREF($3);
		}

reference
	: IDENTIFIER "->" IDENTIFIER	{
			PyObject* ref = py_reference(yyscanner, Py_None, $1, @1);
			$$ = py_reference(yyscanner, ref, $3, @3);
			free($1);
			free($3);
			Py_XDECREF(ref);
		}
	| reference "->" IDENTIFIER	{
			$$ = py_reference(yyscanner, $1, $3, @3);
			free($3);
			Py_XDECREF($1);
		}

clone
	: IDENTIFIER ":" command ";"	{
			$$ = py_clone(yyscanner, $1, $3);
			free($1);
			Py_XDECREF($3);
		}

arguments
	: /* empty */			{ $$ = PyList_New(0); }
	| arguments "," argument	{
			if ($3 != Py_None) PyList_Append($1, $3);
			Py_XDECREF($3);
			$$ = $1;
		}

argument
	: argument_assign		{ $$ = $1; }
	| flag				{ $$ = $1; }
	| error				{ $$ = Py_None; }

flag
	: IDENTIFIER			{
			$$ = py_assign(yyscanner, $1, Py_True);
			free($1);
		}
	| "-" IDENTIFIER		{
			$$ = py_assign(yyscanner, $2, Py_False);
			free($2);
		}

argument_assign
	: IDENTIFIER "=" array	{
			$$ = py_assign(yyscanner, $1, $3);
			free($1); Py_XDECREF($3);
		}
	| IDENTIFIER "=" sum		{
			$$ = py_assign(yyscanner, $1, $3);
			free($1); Py_XDECREF($3);
		}
	| IDENTIFIER "=" boolean	{
			$$ = py_assign(yyscanner, $1, $3);
			free($1);
		}
	| IDENTIFIER "=" STRING_LITERAL	{
			$$ = py_assign(yyscanner, $1, py_string(yyscanner, $3));
			free($1);
			free($3);
		}

line
	: line_body "endbeamline" ";"	{
			py_end_beamline(yyscanner, $1);
			Py_XDECREF($1);
		}

line_body
	: line_body clone		{
			py_add_element(yyscanner, $1, $2);
			Py_XDECREF($2);
			$$ = $1;
		}
	| line_commands		{ $$ = $1; }

line_commands
	: line_commands command ";"		{
			py_add_command(yyscanner, $1, $2, @$);
			Py_XDECREF($2);
			$$ = $1;
		}
	| line_head			{ $$ = $1; }

line_head
	: IDENTIFIER ":" "beamline" arguments ";"	{
			$$ = py_start_beamline(yyscanner, $1, $4, @$);
			free($1);
			Py_XDECREF($4);
		}

command
	: IDENTIFIER arguments  	{
			$$ = py_command(yyscanner, $1, $2, @$);
			free($1);
			Py_XDECREF($2);
		}

array
	: "{" scalar_list "}" { $$ = $2; }
	| "{" error "}"  {
			// Recover from a bad array.
			$$ = PyList_New(0);
		}

scalar_list
	: sum				{
			$$ = PyList_New(0);
			PyList_Append($$, $1);
		}
	| scalar_list "," sum		{
			PyList_Append($1, $3);
			$$ = $1;
			Py_XDECREF($3);
		}

sum
	: product			{ $$ = $1; }
	| sum "+" product		{
			$$ = py_binary_op(yyscanner, "add", $1, $3);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}
	| sum "-" product		{
			$$ = py_binary_op(yyscanner, "sub", $1, $3);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}

product
	: power				{ $$ = $1; }
	| product "*" power		{
			$$ = py_binary_op(yyscanner, "mul", $1, $3);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}
	| product "/" power		{
			$$ = py_binary_op(yyscanner, "truediv", $1, $3);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}
	| product "%" power		{
			$$ = py_binary_op(yyscanner, "mod", $1, $3);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}

power
	: atom				{ $$ = $1; }
	| power "^" atom		{
			$$ = py_binary_op(yyscanner, "pow", $1, $3);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}

atom
	: FLOAT				{ $$ = py_float(yyscanner, $1); }
	| INTEGER			{ $$ = py_integer(yyscanner, $1); }
	| "-" atom			{
			$$ = py_unary_op(yyscanner, "neg", $2);
			Py_XDECREF($2);
		}
	| "+" atom			{ $$ = $2; }
	| "(" sum ")"			{ $$ = $2; }
	| IDENTIFIER			{
			$$ = py_identifier_atom(yyscanner, $1, @1);
			free($1);
		}
	| IDENTIFIER "(" sum ")"	{
			$$ = py_call_func(yyscanner, $1, $3);
			free($1);
			Py_XDECREF($3);
		}
	| "(" error ")"  		{
			// Recover from an error in brackets.
			$$ = py_float(yyscanner, NAN);
		}

boolean
	: TRUE				{ $$ = Py_True; }
	| FALSE				{ $$ = Py_False; }

%%
