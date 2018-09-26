# Compiler to use. in C++, so g++ is used.
CXX=g++

# Libraries to include (Change to local directory would probably be better)
INC= 

# Compiler flags. If desired add -g for debugging info.
CFLAGS= -g -std=c++11 -Wall -Wextra -fopenmp -lm -ffast-math -funroll-loops -O3

# Target executable
TARGET = WCXSPH

#all : $(TARGET) 

$(TARGET) : WCXSPH.cpp
	$(CXX) $(INC) $(CFLAGS) -o $(TARGET) $< 

clean:
	$(RM) $(TARGET)

new:
	$(RM) $(TARGET)
	$(CXX) $(INC) $(CFLAGS) -o $(TARGET) $< 

3D	: WCXSPH3D.cpp
	$(CXX) $(INC) $(CFLAGS) -o $(TARGET) $< 

linux: $(TARGET)
	./WCSPH

win: $(TARGET)
	./$(TARGET).exe

