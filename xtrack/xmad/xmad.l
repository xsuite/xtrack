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

#include "xmad_tab.h"
#include "xmad.h"

// A common macro to keep track of token locations in `yylloc`
#define YY_USER_ACTION                                                 \
    yylloc->first_line = yylloc->last_line;                            \
    yylloc->first_column = yylloc->last_column;                        \
    if (yylloc->last_line == yylineno)                                 \
        yylloc->last_column += yyleng;                                 \
    else {                                                             \
        yylloc->last_line = yylineno;                                  \
        yylloc->last_column = yytext + yyleng - strrchr(yytext, '\n'); \
    }

// Parse the current token value (in `yytext`) as a floating point value
double parse_numeric(char* text, char* error) {
    char* end_ptr;
    double value = strtod(text, &end_ptr);
    if (end_ptr == text) {
        *error = 1;
    }
    return value;
}

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

    /* Define what identifiers and number literals look like */
NUMERIC             [0-9]*(\.[0-9]|[0-9]\.|[0-9])[0-9]*(e[+-]?[0-9]+)?
IDENTIFIER          [A-Za-z_$][A-Za-z0-9_\.$]*
STRING_LITERAL      \"([^\n\"]|\\\")*\"

%%

    /* Basic syntax */
[ \t\n]+            {}
"("                 { return PAREN_OPEN; }
")"                 { return PAREN_CLOSE; }
"{"                 { return BRACE_OPEN; }
"}"                 { return BRACE_CLOSE; }
"sequence"          { return STARTSEQUENCE; }
"endsequence"       { return ENDSEQUENCE; }
":"                 { return COLON; }
","                 { return COMMA; }
";"                 { return SEMICOLON; }

    /* Values */
{NUMERIC}           {
        char error = 0;
        yylval->number = parse_numeric(yytext, &error);
        if (error) {
            yyerror(yylloc, yyscanner, "an invalid numeric value encountered");
        }
        return FLOAT;
    }
{IDENTIFIER}        {
        yylval->string = strdup(yytext);
        return IDENTIFIER;
    }

    /* Assignments */
"="                 { return ASSIGN; }
":="                { return WALRUS; }

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
"\""                BEGIN(STRING_STATE);
<STRING_STATE>"\""  {
        BEGIN(INITIAL);
        yylval->string = strndup(yytext, yyleng - 1);
        return STRING_LITERAL;
    }
<STRING_STATE>"\\\"" { yymore(); }
<STRING_STATE>"\n"  {
        // Let's disallow line breaks in strings. Emit error, and try to recover.
        yyerror(yylloc, yyscanner, "unfinished string literal");
        BEGIN(INITIAL);
    }
<STRING_STATE>[^\"] { yymore(); }

    /* Handle invalid input */
.                   { BEGIN(LEXER_ERROR); yyless(1); }
<LEXER_ERROR>.+     {
        // Emits an error, gobbles the rest of the line, and attempts recovery
        yyerror(yylloc, yyscanner, "tokeniser encountered invalid input");
        BEGIN(INITIAL);
    }

%%
