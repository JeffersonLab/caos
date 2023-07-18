

OBJ := obj

SOURCES := $(wildcard *.c)
OBJECTS := $(patsubst %.c, $(OBJ)/%.o, $(SOURCES))

INCLUDES=-I$(JVM_HOME)/include
FLAGS=-Wformat

all: $(OBJECTS)
	ar rc libevio.a $^

clean:
	@echo cleaning distribution
	@rm -rf *~ *.o
	@rm -rf obj

$(OBJ)/%.o: %.c
	@test -d obj || mkdir -p obj
	$(CC) $(FLAGS) -c $< -o $@ -I.
