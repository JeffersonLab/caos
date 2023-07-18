

exec: et_start.o
	$(CC) -o et_start $< ../libet.a

clean:
	@echo cleaning distribution
	@rm -rf *~ *.o
	@rm -rf obj

%.o: %.c
	$(CC) -c $< -o $@ -I. -I../lib
