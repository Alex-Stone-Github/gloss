.PHONY: clean run

bin/test: src/example.c bin/libgloss.so
	gcc $< -o $@ -L./bin -Wl,-rpath=./bin -lgloss -I./include

bin/libgloss.so: build/graph.o
	gcc $^ -o $@ -shared -fPIC

build/%.o: src/%.c
	gcc -c $^ -o $@ -I./include

clean:
	rm bin/*
	rm build/*

run:
	./bin/test
