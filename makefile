CXX=g++
LIBS = -I /usr/local/include/eigen3 -I /usr/local/include
CFLAGS=-g -fbounds-check -std=c++11 -O3 -fstrict-aliasing 

WCSPH : WCSPH_XSPH.cpp
	$(CXX) $(LIBS) $(CFLAGS) -o $@ $<

linux: WCSPH
	./WCSPH Sample_Input.txt

win: WCSPH
	./WCSPH.exe Sample_Input.txt
