CXX=g++
CFLAGS=-g -fbounds-check -std=c++11 -O4 -fstrict-aliasing 

SPH_Generator : SPH_Generator_single.cpp
	$(CXX) $(CFLAGS) -o  $@ $<

test: SPH_Generator
	./SPH_Generator Sample_Input.txt

bench: SPH_Generator_Single
	./SPH_Generator_Single Sample_Input.txt
