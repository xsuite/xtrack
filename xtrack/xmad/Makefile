CC=clang
CFLAGS=$(shell python3-config --cflags)
LDFLAGS=$(shell python3-config --ldflags)

.PHONY: clean

all: xmad.c xmad_tab.c xmad_lex.c
	cd ../..; python setup.py build_ext --inplace
#	$(CC) $(CFLAGS) -c xmad.c xmad_tab.c xmad_lex.c
#	$(CC) $(LDFLAGS) *.o -o xmad

parser: xmad_tab.c xmad_lex.c

xmad.c: xmad.py xmad.pxd
	cython xmad.py --embed -o xmad.c

xmad_tab.c: xmad.y
	bison xmad.y -t -k -d -Wcounterexamples -o xmad_tab.c

xmad_lex.c: xmad.l xmad_tab.h lex_helpers.h
	flex -d --header-file=xmad_lex.h -o xmad_lex.c xmad.l

clean:
	rm xmad_tab.* xmad_lex.* xmad.c *.o
