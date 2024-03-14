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
 `yytext` pointer outside the *.l file, as it's lifetime is generally only
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

#include "xmad_tab.h"
#include "xmad_lex.h"
#include "xmad.h"
%}

%lex-param {void* yyscanner}
%parse-param {void* yyscanner}

// The union type that store token values
%union {
    double number;
    char* string;
    PyObject* object;
}

// Basic syntax
%token PAREN_OPEN		"("
%token PAREN_CLOSE		")"
%token BRACE_OPEN		"{"
%token BRACE_CLOSE		"}"
%token STARTSEQUENCE		"startsequence"
%token ENDSEQUENCE		"endsequence"
%token COLON			":"
%token COMMA			","
%token SEMICOLON		";"
// Values
%token<number> FLOAT		"number"
%token<string> IDENTIFIER	"identifier"
%token<string> STRING_LITERAL	"string literal"
// Assignments
%token ASSIGN			"="
%token WALRUS			":="
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
%type <object> clone argument
%type <object> eq_defer eq_value flag eq_keyword eq_value_scalar eq_defer_scalar
%type <object> atom power product sum
%type <object> arguments elements array scalar_list

// Clean up token values on error
%destructor { free($$); } IDENTIFIER STRING_LITERAL

// Associativity rules
%left ADD SUB
%left MUL DIV MOD
%left EQ NE GT GE LT LE

%%

start
	: statement
	| start statement

statement
	: set_defer
	| set_value
	| command_stmt
	| clone
	| sequence
	| error SEMICOLON  // Recover from an erroneous line.

set_defer
	: eq_defer_scalar SEMICOLON	{ py_set_defer(yyscanner, $1); }

set_value
	: eq_value_scalar SEMICOLON	{ py_set_value(yyscanner, $1); }

clone
	: IDENTIFIER COLON IDENTIFIER arguments SEMICOLON	{
			$$ = py_clone($1, $3, $4);
			free($1);
			free($3);
		}

command_stmt
	: IDENTIFIER arguments SEMICOLON  { free($1); }  // TODO

arguments
	: /* empty */			{ $$ = PyList_New(0); }
	| COMMA argument arguments	{ PyList_Append($3, $2); $$ = $3; }

argument
	: eq_defer 			{ $$ = $1; }
	| eq_value			{ $$ = $1; }
	| flag				{ $$ = $1; }
	| eq_keyword			{ $$ = $1; }

flag
	: ADD IDENTIFIER		{ $$ = py_eq_value_scalar($2, Py_True); free($2); }
	| SUB IDENTIFIER		{ $$ = py_eq_value_scalar($2, Py_False); free($2); }

eq_keyword
	: IDENTIFIER ASSIGN STRING_LITERAL	{
			$$ = py_eq_value_scalar($1, PyUnicode_FromString($3));
			free($1);
			free($3);
		}

eq_value_scalar
	: IDENTIFIER ASSIGN sum		{ $$ = py_eq_value_scalar($1, $3); free($1); }

eq_defer_scalar
	: IDENTIFIER WALRUS sum		{ $$ = py_eq_defer_scalar($1, $3); free($1); }

eq_value
	: IDENTIFIER ASSIGN array	{ $$ = py_eq_value_array($1, $3); free($1); }
	| IDENTIFIER ASSIGN sum		{ $$ = py_eq_value_sum($1, $3); free($1); }

eq_defer
	: IDENTIFIER WALRUS array	{ $$ = py_eq_defer_array($1, $3); free($1); }
	| IDENTIFIER WALRUS sum		{ $$ = py_eq_defer_sum($1, $3); free($1); }

sequence
	: IDENTIFIER COLON STARTSEQUENCE arguments SEMICOLON
	  elements
	  ENDSEQUENCE SEMICOLON		{
	  		py_make_sequence(yyscanner, $1, $4, $6);
	  		free($1);
	  	}

elements
	: /* empty */			{ $$ = PyList_New(0); }
	| elements clone		{ PyList_Append($1, $2); $$ = $1; }
	| error SEMICOLON  		{
			// Recover from a bad line without breaking the
			// sequence (otherwise falls back to `statement`).
			$$ = Py_None;
		}

array
	: BRACE_OPEN scalar_list BRACE_CLOSE { $$ = $2; }
	| BRACE_OPEN error BRACE_CLOSE  {
			// Recover from a bad array.
			$$ = PyList_New(0);
		}

scalar_list
	: sum				{ $$ = PyList_New(0); PyList_Append($$, $1); }
	| scalar_list COMMA sum		{ PyList_Append($1, $3); $$ = $1; }

sum
	: product			{ $$ = $1; }
	| sum ADD product		{ $$ = py_binary_op("add", $1, $3); }
	| sum SUB product		{ $$ = py_binary_op("sub", $1, $3); }

product
	: power				{ $$ = $1; }
	| product MUL power		{ $$ = py_binary_op("mul", $1, $3); }
	| product DIV power		{ $$ = py_binary_op("truediv", $1, $3); }
	| product MOD power		{ $$ = py_binary_op("mod", $1, $3); }

power
	: atom				{ $$ = $1; }
	| power POW atom		{ $$ = py_binary_op("pow", $1, $3); }

atom
	: FLOAT				{ $$ = py_float($1); }
	| SUB atom			{ $$ = py_unary_op("neg", $2); }
	| ADD atom			{ $$ = $2; }
	| PAREN_OPEN sum PAREN_CLOSE	{ $$ = $2; }
	| IDENTIFIER			{ $$ = py_identifier_atom(yyscanner, $1); free($1); }
	| IDENTIFIER ARROW IDENTIFIER	{ $$ = py_arrow(yyscanner, $1, $3); free($1); free($3); }
	| IDENTIFIER PAREN_OPEN sum PAREN_CLOSE	{ $$ = py_call_func(yyscanner, $1, $3); free($1); }
	| PAREN_OPEN error PAREN_CLOSE  {
			// Recover from an error in brackets.
			$$ = py_float(NAN);
		}


%%
