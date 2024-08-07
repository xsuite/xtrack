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
%token STARTLINE		"beamline"
%token ENDLINE			"endbeamline"
%token COLON			":"
%token COMMA			","
%token SEMICOLON		";"
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
%type <object> clone argument start_line start_line_modern
%type <object> argument_assign flag variable_assign command
%type <object> atom power product sum reference
%type <object> arguments elements commands array scalar_list

// Clean up token values on error
%destructor { free($$); } IDENTIFIER STRING_LITERAL
// Clean up the python lists we create as part of grammar actions
%destructor { Py_DECREF($$); } arguments array scalar_list elements commands

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
	: IDENTIFIER ":" IDENTIFIER arguments ";"	{
			$$ = py_clone(yyscanner, $1, $3, $4);
			free($1);
			free($3);
			Py_XDECREF($4);
		}

arguments
	: /* empty */			{ $$ = PyList_New(0); }
	| arguments "," argument	{
			PyList_Append($1, $3);
			$$ = $1;
			Py_XDECREF($3);
		}

argument
	: argument_assign		{ $$ = $1; }
	| flag				{ $$ = $1; }

flag
	: "+" IDENTIFIER		{
			$$ = py_assign(yyscanner, $2, Py_True);
			free($2);
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
	| IDENTIFIER "=" STRING_LITERAL	{
			$$ = py_assign(yyscanner, $1, PyUnicode_FromString($3));
			free($1);
			free($3);
		}

line
	: start_line commands elements "endbeamline" ";"	{
			py_make_sequence(yyscanner, $1);
			// Unfortunately we must handle the reference counting
			// of the LineTemplate object here: it is leaving the
			// "C scope" and we don't need it anymore.
			Py_XDECREF($1);
			Py_XDECREF($3);
		}
	| start_line_modern commands elements "}" ";" 	{
			py_make_sequence(yyscanner, $1);
			Py_XDECREF($1);
			Py_XDECREF($3);
		}

start_line
	: IDENTIFIER ":" "beamline" arguments ";"	{
			$$ = py_start_sequence(yyscanner, $1, $4, @$);
			free($1);
			Py_XDECREF($4);
		}

start_line_modern
	: IDENTIFIER ":" "beamline" arguments "{"	{
			$$ = py_start_sequence(yyscanner, $1, $4, @$);
			free($1);
			Py_XDECREF($4);
		}

elements
	: clone				{
			// $<object>0 is the LineTemplate object taken from the
			// top of the stack, which is the result of `start_line`
			$$ = py_new_element(yyscanner, $<object>0, $1);
			// Handling the reference counting, see `sequence`
			Py_XDECREF($$);
			Py_XDECREF($1);
		}
	| elements clone		{
			$$ = py_new_element(yyscanner, $1, $2);
			// Handling the reference counting, see `sequence`
			Py_XDECREF($1);
			Py_XDECREF($2);
		}
	| error SEMICOLON  		{
			// Recover from a bad line without breaking the
			// sequence (otherwise falls back to `statement`).
			$$ = $<object>0;
		}

commands
	: /* empty */ 			{ $$ = $<object>0; }
	| commands command	{
			// $<object>0 is the LineTemplate object, see `elements`
			py_add_command(yyscanner, $1, $2, @$);
			Py_XDECREF($2);
			$$ = $1;
		}

command
	: IDENTIFIER arguments ";"  	{
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

%%
