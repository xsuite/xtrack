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

#define YY_USER_ACTION \
    yylloc.first_line = yylloc.last_line; \
    yylloc.first_column = yylloc.last_column; \
    for(int i = 0; yytext[i] != '\0'; i++) { \
        if(yytext[i] == '\n') { \
            yylloc.last_line++; \
            yylloc.last_column = 0; \
        } \
        else { \
            yylloc.last_column++; \
        } \
    }

extern int yyerror(const char*);

double parse_numeric() {
    char* err;
    double value = strtod(yytext, &err);
    if (err == yytext) {
        yyerror("tokeniser found an invalid numeric value");
    }
    return value;
}

%}

%option noyywrap
%option yylineno

%x COMMENT
%x LEXER_ERROR

NUMERIC             [0-9]*(\.[0-9]|[0-9]\.|[0-9])[0-9]*(e[+-]?[0-9]+)?
IDENTIFIER          [A-Za-z_$][A-Za-z0-9_\.$]*

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
        yylval.number = parse_numeric();
        return FLOAT;
    }
{IDENTIFIER}        {
        yylval.string = strdup(yytext);
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


    /* Handle invalid input */
.                   { BEGIN(LEXER_ERROR); yyless(1); }
<LEXER_ERROR>.+     { yyerror("tokeniser encountered invalid input"); }

%%
