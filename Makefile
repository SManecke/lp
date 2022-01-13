all:
	mkdir -p build
	gcc -g -O0 -Wall lp.c csparse.c -o build/lp -lm
fmt:
	find . -regex '.*\.\(c\|h\)' -exec clang-format -style=file -i {} \;
