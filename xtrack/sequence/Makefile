CC=clang
CFLAGS=$(shell python3-config --cflags)
LDFLAGS=$(shell python3-config --ldflags)

.PHONY: clean

all: parser.c parser_tab.c lexer.c
	cd ../..; python setup.py build_ext --inplace
#	$(CC) $(CFLAGS) -c xmad.c xmad_tab.c xmad_lex.c
#	$(CC) $(LDFLAGS) *.o -o xmad

parser: parser_tab.c lexer.c

parser.c: parser.py parser.pxd
	cython parser.py --embed -o parser.c

parser_tab.c: grammar.y
	bison grammar.y -t -k -d -Wcounterexamples -o parser_tab.c

lexer.c: lexer.l parser_tab.h lex_helpers.h
	flex -d --header-file=lexer.h -o lexer.c lexer.l

clean:
	rm parser_tab.c parser_tab.h lexer.c lexer.h parser.c parser.h *.o *.so
