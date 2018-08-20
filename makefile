CXX=g++
INC= -I C:\cygwin64\usr\include\eigen3 -I C:\cygwin64\usr\local\include 
CFLAGS= -g -fbounds-check -std=c++11 -O3 -fstrict-aliasing -o

WCSPH : WCSPH_XSPH.cpp
	$(CXX) $(INC) $(CFLAGS) $@ $< 

linux: WCSPH
	./WCSPH Sample_Input.txt

win: WCSPH
	./WCSPH.exe Sample_Input.txt

