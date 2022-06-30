CC=gcc
IFLAG=
LINKER= -lm -lpcap
PROFILER=

pcapParser: main.o
	$(CC) -o pcapParser compiled/main.o $(LINKER) $(PROFILER)

main.o:
	$(CC) -c -Wall main.c -o compiled/main.o$(IFLAG) $(PROFILER)

clean:
	rm -f pcapParser
	rm -f compiled/*.o