CC=gcc
CFLAGS=-Iinclude/ -g
DEPS = $(wildcard include/*.h)
OBJ = ehr.o list.o patient.o

%.o: src/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

ehr: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

ehr_tests: tests/ehr_tests.c
	$(CC) -o $@ $^ $(CFLAGS) -lcunit -g

.PHONY: clean

clean:
	rm -rf ehr ehr_tests *.o
