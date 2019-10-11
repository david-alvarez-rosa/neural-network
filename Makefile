# Este Makefile te servirá para compilar al menos el 90% de tus
# proyectos en C/C++

# Referencias: http://make.mad-scientist.net/papers/
# advanced-auto-dependency-generation/

# Puedes cambiar los valores de las variables PRN,
# OBJDIR, DEPDIR:

# PRN=nombre_del_ejecutable
# OBJDIR=nombre_directorio donde se escribirán los ficheros .o
# DEPDIR=nombre_directorio donde se escribirán los ficheros .d

PRN=main
OBJDIR=.o
DEPDIR=.d

# Uso:

# para compilar:
# ~$ make

# para borrar los .o:
# ~$ make clean

# para borrar también el ejecutable:
# ~$ make mrproper

EXE=$(PRN:=.exe)
$(shell mkdir -p $(OBJDIR) >/dev/null)
$(shell mkdir -p $(DEPDIR) >/dev/null)

SRCS_ALL=$(wildcard *.cc)
SRCS=$(filter-out %_flymake.cc, $(SRCS_ALL))
OBJS=$(patsubst %,$(OBJDIR)/%.o,$(basename $(SRCS)))
DEPS=$(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))

CFLAGS=
CXXFLAGS= -g -O2 --std=c++11
CPPFLAGS += -MMD -MP -MF $(DEPDIR)/$*.Td
LIBS=
LDFLAGS=

POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

$(PRN): $(OBJS)
	$(CXX) -o$@ $^ $(LDFLAGS) $(LIBS)
	@echo
	@echo '"make help"', para ver otras opciones
	@echo

help:
	@echo
	@echo '"make"', para compilar el programa
	@echo '"make clean"', para borrar los .o \(y .d\)
	@echo '"make mrproper"', bara borrar también los ejecutables
	@echo

clean:
	rm -f $(OBJS) $(DEPS) *~
	rmdir $(OBJDIR) $(DEPDIR)

mrproper: clean
	rm -f $(PRN) $(EXE)

$(OBJDIR)/%.o: %.cc $(DEPDIR)/%.d
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o$@
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(DEPS)

.PHONY: clean mrproper
