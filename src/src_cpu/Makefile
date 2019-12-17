#### User configurable options #####
include ../Makefile.config

.SUFFIXES: 
.SUFFIXES: .cpp .cu .c .o

EXEC = SeisAcoMod2D_MULTI_CORE_CPU_V1.0.exe

Objects = main.o json_parser.o modelling_worker.o modelling_master.o read_receiver.o create_geometry.o \
          workload.o model.o check_stability.o cpu_modelling_kernels.o cpumodelling.o

all : $(EXEC)

$(EXEC):$(Objects)
	$(CPPLINKER) $(FFLAGS) -o $(EXEC) $(Objects) $(FLIBS) $(FFLAGS)
	mv $(EXEC) ../../bin/

clean:
	rm -f *.o *~

.cpp.o:
	$(CPPC) $(FFLAGS) -c $*.cpp $(FLIBS) $(FFLAGS)

cleano:
	rm -f *.o *~	
