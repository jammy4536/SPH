CXX=g++
INC= -I C:\cygwin64\usr\include\eigen3 -I C:\cygwin64\usr\local\include
CFLAGS= -g -fbounds-check -fstrict-aliasing -std=c++11 -O3  -o
exe = WCSPH

XSPH : WCXSPH.cpp
	$(CXX) $(INC) $(CFLAGS) $(exe) $< 

SPH : WCSPH.cpp
	$(CXX) $(INC) $(CFLAGS) $(exe) $< 

3D	: WCXSPH3D.cpp
	$(CXX) $(INC) $(CFLAGS) $(exe) $< 

linux: WCSPH
	./WCSPH Sample_Input.txt

win: WCSPH
	$(CXX) $(INC) $(CFLAGS) $(exe) WCXSPH.cpp 
	./WCSPH.exe Sample_Input.txt

clean:
	rm WCSPH WCSPH.exe
