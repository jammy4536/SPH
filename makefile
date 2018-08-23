CXX=g++
INC= -I C:\cygwin64\usr\include\eigen3 -I C:\cygwin64\usr\local\include 
CFLAGS= -g -fbounds-check -std=c++11 -O3 -fstrict-aliasing -o
exe = WCSPH

SPH : WCSPH.cpp
	$(CXX) $(INC) $(CFLAGS) $(exe) $< 

XSPH : WCXSPH.cpp
	$(CXX) $(INC) $(CFLAGS) $(exe) $< 

3D	: WCXSPH3D.cpp
	$(CXX) $(INC) $(CFLAGS) $(exe) $< 

linux: WCSPH
	./WCSPH Sample_Input.txt

win: WCSPH
	./WCSPH.exe Sample_Input.txt
