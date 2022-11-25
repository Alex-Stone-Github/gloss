SRC_FILES = $(wildcard src/*.cpp);
EXT_FILES = $(subst .cpp,.o,$(SRC_FILES))
OBJ_FILES = $(subst src,build,$(EXT_FILES))

.PHONY: run clean

build/out: $(OBJ_FILES)
	g++ $^ -o $@
build/%.o: src/%.cpp
	g++ -c $^ -o $@ -I./include

run: build/out
	./build/out
clean:
	rm ./build/*
