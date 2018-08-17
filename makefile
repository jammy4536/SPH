CXX=g++
CFLAGS=-g -fbounds-check -std=c++11 -O3 -fstrict-aliasing 

WCSPH : WCSPH_XSPH.cpp
	$(CXX) $(CFLAGS) -o  $@ $<

lin: WCSPH
	./WCSPH Sample_Input.txt

win: WCSPH
	./WCSPH.exe Sample_Input.txt
