#### User configurable options #####
include ../Makefile.config 

.SUFFIXES: 
.SUFFIXES: .cpp .cu .c .o

EXEC = SeisAcoMod2D_GPU_OPENACC_V1.0.exe

Objects = main.o json_parser.o modelling_worker.o modelling_master.o read_receiver.o create_geometry.o \
          workload.o model.o check_stability.o cpu_modelling_kernels.o cpumodelling.o

all : $(EXEC)

$(EXEC):$(Objects)
	$(PGI_CPPLINKER) $(PGI_FFLAGS) -o $(EXEC) $(Objects) $(PGI_FLIBS) $(PGI_FFLAGS)
	mv $(EXEC) ../../bin/

clean:
	rm -f *.o *~

.cu.o:
	nvcc $(PGI_NVFLAGS) -c $*.cu

.cpp.o:
	$(PGI_CPPC) $(PGI_FFLAGS) -c $*.cpp $(PGI_FLIBS) $(PGI_FFLAGS)

cleano:
	rm -f *.o *~	
