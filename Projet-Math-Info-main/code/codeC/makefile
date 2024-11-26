CC = gcc
EXC = main
OBJDIR = obj
SRCDIR = src
SRC = $(wildcard $(SRCDIR)/*.c)
OBJ = $(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

all: $(EXC)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -o $@ -c $< -lm
 
$(OBJDIR)/main.o: main.c
	$(CC) -o $@ -c $< -lm

$(EXC): $(OBJ) $(OBJDIR)/main.o
	echo "Compilation avec succes !, Pour executer tapez $(EXC)"
	$(CC) -o $@ $^ -lm
	echo "Pour executer LU_efficace.c, faut le faire avec la commande gcc !"

clean:
	rm -rf $(OBJDIR)/*.o $(EXC)

.PHONY: clean all