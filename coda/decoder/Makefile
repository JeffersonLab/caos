#********************************************************
# MAKEFILE for BUILDING EXAMPLES FOR HIPO4 LIBRARY
# AUTHOR: GAVALIAN DATE: 10/24/2018
#********************************************************

ifeq ($(HIPODIR),)
abort:   ## This MUST be the first target :( ugly
	@echo 
	@echo 'environment HIPODIR is not set. please define it'
	@echo 'this is the way... \n\n' && false
endif

HIPOCFLAGS  := -I$(HIPODIR)/hipo4 
HIPOLIBS    := -L$(HIPODIR)/lib -lhipo4

LZ4LIBS     := -L$(HIPODIR)/lz4/lib -llz4
LZ4INCLUDES := -I$(HIPODIR)/lz4/lib

CXX       := g++
CC        := gcc
#CXXFLAGS  += -Wall -fPIC -std=c++11 -Wreorder-ctor
CXXFLAGS  += -w -fPIC -std=c++11 -Wreorder-ctor
LD        := g++
LDFLAGS   :=

OBJECTS = translate.o table.o decoder.o evioBankUtil.o component.o detectors.o fadc250fitter.o

all:  translate

translate: $(OBJECTS)
	$(CXX) -o translate.exe $(OBJECTS) $(HIPOLIBS) $(LZ4LIBS)

#analysis: analysis.o
#	$(CXX) -o analysis $<   $(HIPOLIBS) $(LZ4LIBS)

#clas12event_example: clas12event_example.o
#	$(CXX) -o clas12event_example $<   $(HIPOLIBS) $(LZ4LIBS)

clean:
	@echo 'Removing all build files'
	@rm -rf *.o *~ *.exe example*hipo 

%.o: %.cc
	$(CXX) -c $< -O3 $(CXXFLAGS) $(HIPOCFLAGS) $(LZ4INCLUDES)
%.o: %.c
	$(CC) -c $< -O3 $(HIPOCFLAGS) $(LZ4INCLUDES)
