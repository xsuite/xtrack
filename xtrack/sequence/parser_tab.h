/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Skeleton interface for Bison GLR parsers in C

   Copyright (C) 2002-2015, 2018-2021 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_PARSER_TAB_H_INCLUDED
# define YY_YY_PARSER_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    PAREN_OPEN = 258,              /* "("  */
    PAREN_CLOSE = 259,             /* ")"  */
    BRACE_OPEN = 260,              /* "{"  */
    BRACE_CLOSE = 261,             /* "}"  */
    COLON = 262,                   /* ":"  */
    COMMA = 263,                   /* ","  */
    SEMICOLON = 264,               /* ";"  */
    STARTLINE = 265,               /* "beamline"  */
    ENDLINE = 266,                 /* "endbeamline"  */
    TRUE = 267,                    /* "true"  */
    FALSE = 268,                   /* "false"  */
    FLOAT = 269,                   /* "floating point number"  */
    INTEGER = 270,                 /* "integer number"  */
    IDENTIFIER = 271,              /* "identifier"  */
    STRING_LITERAL = 272,          /* "string literal"  */
    ASSIGN = 273,                  /* "="  */
    EQ = 274,                      /* "=="  */
    NE = 275,                      /* "!="  */
    GT = 276,                      /* ">"  */
    GE = 277,                      /* ">="  */
    LT = 278,                      /* "<"  */
    LE = 279,                      /* "<="  */
    ADD = 280,                     /* "+"  */
    SUB = 281,                     /* "-"  */
    MUL = 282,                     /* "*"  */
    DIV = 283,                     /* "/"  */
    MOD = 284,                     /* "%"  */
    POW = 285,                     /* "^"  */
    ARROW = 286                    /* "->"  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 74 "grammar.y"

    double floating;
    long integer;
    char* string;
    PyObject* object;

#line 97 "parser_tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif



int yyparse (void* yyscanner);

#endif /* !YY_YY_PARSER_TAB_H_INCLUDED  */
