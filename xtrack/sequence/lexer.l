/* copyright *********************************
 * This file is part of the Xtrack Package.  *
 * Copyright (c) CERN, 2024.                 *
 *********************************************

 A lexer definition for a MAD-X inspired lattice description language.
 The lexer described in this file does basic processing of the input, and
 spits out tokens that will be interpreted by the accompanying parser.
 The parser is defined in the *.y file, that is where the token definitions are.
*/


%{
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>

#include "parser_tab.h"
#include "parser.h"
#include "lex_helpers.h"

%}

    /* Stop on end of file, and keep track of line numbers */
%option noyywrap yylineno
    /* We don't care for customising lexical analysis (feeding tokens, etc.) */
%option noinput nounput
    /* Make a re-entrant parser (thread safety!) */
%option reentrant bison-bridge bison-locations
    /* Hold a pointer to the python parser object within the scanner */
%option extra-type="PyObject*"

    /* Special exclusive states to enter to skip a bunch of input, see later */
%x COMMENT
%x LEXER_ERROR
%x STRING_STATE

    /* Define what identifiers and number literals look like

      It seems that GCC assumes that any number constant written in engineering
      notation is automatically a double. For lack of a better reason to do it
      differently, let's keep this convention: a sequence of digits -> integer,
      as soon as '.' or 'e' appears, we have a float.
    */
INTEGER             [0-9]+
FLOAT               [0-9]*(\.[0-9]|[0-9]\.)[0-9]*(e[+-]?[0-9]+)?|[0-9]+e[+-]?[0-9]+
    /* Identifiers are fairly usual, but we allow '.' just like MAD-X. */
IDENTIFIER          [A-Za-z_$][A-Za-z0-9_\.$]*

%%

    /* Basic syntax */
[ \t\n]+            {}
"("                 { return PAREN_OPEN; }
")"                 { return PAREN_CLOSE; }
"{"                 { return BRACE_OPEN; }
"}"                 { return BRACE_CLOSE; }
":"                 { return COLON; }
","                 { return COMMA; }
";"                 { return SEMICOLON; }

    /* Keywords */
"beamline"          { return STARTLINE;}
"endbeamline"       { return ENDLINE; }
"true"              { return TRUE; }
"false"             { return FALSE; }

    /* Values */
{FLOAT}             {
        char error = 0;
        yylval->floating = parse_float(yytext, &error);
        if (error) {
            yyerror(yylloc, yyscanner, "an floating point value encountered");
        }
        return FLOAT;
    }
{INTEGER}             {
        char error = 0;
        yylval->integer = parse_integer(yytext, &error);
        if (error) {
            yyerror(yylloc, yyscanner, "an invalid integer value encountered");
        }
        return INTEGER;
    }
{IDENTIFIER}        {
        yylval->string = strdup(yytext);
        return IDENTIFIER;
    }

    /* Assignments */
"="                 { return ASSIGN; }

    /* Comparisons */
"=="                { return EQ; }
"!="                { return NE; }
">"                 { return GT; }
">="                { return GE; }
"<"                 { return LT; }
"<="                { return LE; }

    /* Operators */
"+"                 { return ADD; }
"-"                 { return SUB; }
"*"                 { return MUL; }
"/"                 { return DIV; }
"%"                 { return MOD; }
"^"                 { return POW; }
"**"                { return POW; }

    /* Accessor */
"->"                { return ARROW; }

    /* Comments */
[!#].+              {}
"//".+              {}
"/*"                BEGIN(COMMENT);
<COMMENT>"*/"       BEGIN(INITIAL);
<COMMENT>[^\n]      {}
<COMMENT>"\n"       {}

    /* Strings

       This way of handling strings is a bit convoluted, but is night and day
       in terms of error recovery when compared to a single regex solution.
    */
"\""                {
        BEGIN(STRING_STATE);
        yymore();
        yylloc->last_column -= yyleng;
    }
<STRING_STATE>"\""  {
        BEGIN(INITIAL);
        yylval->string = strndup(yytext + 1, yyleng - 2);
        return STRING_LITERAL;
    }
<STRING_STATE>"\\\"" {
        yymore();
        yylloc->last_column -= yyleng;
    }
<STRING_STATE>"\n"  {
        // Let's disallow line breaks in strings. Emit error, and try to recover.
        yyerror(yylloc, yyscanner, "unfinished string literal");
        BEGIN(INITIAL);
        // This will probably be garbled nonsense, but let's return the string
        // token anyway: the parser will emit a more meaningful error.
        yylval->string = strndup(yytext + 1, yyleng - 2);
        return STRING_LITERAL;
    }
<STRING_STATE>[^\"] {
        yymore();
        yylloc->last_column -= yyleng;
    }

    /* Handle invalid input */
.                   { BEGIN(LEXER_ERROR); yyless(1); }
<LEXER_ERROR>.+     {
        // Emits an error, gobbles the rest of the line, and attempts recovery
        yyerror(yylloc, yyscanner, "tokeniser encountered invalid input");
        BEGIN(INITIAL);
    }

%%
