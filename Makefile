CXX = g++

CXXFLAGS = -std=c++14 -O3 -ffast-math -D CPU_ONLY

LDFLAGS = `pkg-config --libs opencv` -lopencv_face

build/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)

normalize.out: build/normalize.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f build/*.o *.out
