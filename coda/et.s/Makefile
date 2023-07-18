
SRC := lib
OBJ := obj

SOURCES := $(wildcard $(SRC)/*.c)
OBJECTS := $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SOURCES))

INCLUDES=-I$(JVM_HOME)/include
FLAGS=-Wformat

lib: $(OBJECTS)
	ar rc libet.a $^

clean:
	@echo cleaning distribution
	@rm -rf *~ *.o libet.a
	@rm -rf obj


$(OBJ)/%.o: $(SRC)/%.c
	@test -d obj || mkdir -p obj
	$(CC) $(FLAGS) -c $< -o $@ -I. -Ilib
