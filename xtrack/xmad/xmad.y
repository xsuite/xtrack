%{
#include <stdlib.h>
#include <stdio.h>
#include <Python.h>

#define YYERROR_VERBOSE
#include "xmad_lex.h"

extern void yyerror(const char *s);

typedef PyObject* p_obj;
extern p_obj py_float(double);
extern p_obj py_unary_op(const char*, p_obj);
extern p_obj py_binary_op(const char*, p_obj, p_obj);
extern p_obj py_eq_value_scalar(const char*, p_obj);
extern p_obj py_eq_defer_scalar(const char*, p_obj);
extern p_obj py_call_func(const char*, p_obj);
%}

%union {
    double number;
    char* string;
    PyObject* object;
}

// Basic syntax
%token PAREN_OPEN	"("
%token PAREN_CLOSE	")"
%token BRACE_OPEN	"{"
%token BRACE_CLOSE	"}"
%token STARTSEQUENCE	"startsequence"
%token ENDSEQUENCE	"endsequence"
%token COLON		":"
%token COMMA		","
%token SEMICOLON	";"
// Values
%token<number> FLOAT		"number"
%token<string> IDENTIFIER	"identifier"
// Assignments
%token ASSIGN		"="
%token WALRUS		":="
// Comparisons
%token EQ		"=="
%token NE		"!="
%token GT		">"
%token GE		">="
%token LT		"<"
%token LE		"<="
// Operators
// TODO: Logical operators are supported by xdeps (can implement a ternary op.)
%token ADD		"+"
%token SUB		"-"
%token MUL		"*"
%token DIV		"/"
%token MOD		"%"
%token POW		"^"
// Accessor
%token ARROW		"->"

// Nonterminal (rule) types
%type <object> atom power product sum
%type <object> eq_value_scalar eq_defer_scalar

// Associativity rules
%left ADD SUB
%left MUL DIV MOD
%left EQ NE GT GE LT LE

%locations

%%

start
	: statement
	| start statement  // Use left-recursion to minimise stack pressure.
			   // Do this throughout the grammar, if possible.

statement
	: set_defer
	| set_value
	| command_stmt
	| clone
	| sequence
	| error SEMICOLON  // Recover from an erroroneous line.

set_defer
	: eq_defer_scalar SEMICOLON

set_value
	: eq_value_scalar SEMICOLON

clone
	: IDENTIFIER COLON command SEMICOLON

command_stmt
	: command SEMICOLON

command
	: IDENTIFIER arguments_optional

arguments_optional
	: /* empty */
	| arguments

arguments
	: COMMA argument
	| COMMA argument arguments

argument
	: eq_defer | eq_value | flag

flag
	: ADD IDENTIFIER
	| SUB IDENTIFIER

eq_value_scalar
	: IDENTIFIER ASSIGN sum		{ $$ = py_eq_value_scalar($1, $3); }

eq_defer_scalar
	: IDENTIFIER WALRUS sum		{ $$ = py_eq_defer_scalar($1, $3); }

eq_value
	: IDENTIFIER ASSIGN array
	| IDENTIFIER ASSIGN sum

eq_defer
	: IDENTIFIER WALRUS array
	| IDENTIFIER WALRUS sum

sequence
	: IDENTIFIER COLON STARTSEQUENCE arguments_optional SEMICOLON
	  elements
	  ENDSEQUENCE SEMICOLON

elements
	: clone
	| elements clone
	| error SEMICOLON  // Recover from a bad line without breaking the
			   // sequence (otherwise falls back to `statement`).

array
	: BRACE_OPEN scalar_list BRACE_CLOSE
	| BRACE_OPEN error BRACE_CLOSE  // Recover from a bad array.

scalar_list
	: sum
	| sum COMMA scalar_list
	| error COMMA scalar_list  // Recover from an erroneous argument.

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
	| IDENTIFIER			{ $$ = py_float(NAN); }
	| constant			{ $$ = py_float(NAN); }
	| IDENTIFIER ARROW IDENTIFIER	{ $$ = py_arrow($1, $3); }
	| IDENTIFIER PAREN_OPEN sum PAREN_CLOSE	{ $$ = py_call_func($1, $3); }
	| PAREN_OPEN error PAREN_CLOSE  // Recover from an error in brackets.
		{ $$ = py_float(NAN); }

constant
	: "PI"
	| "TWOPI"
	| "CENTRE"

%%
