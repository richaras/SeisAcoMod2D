#### User configurable options #####
include ../../Makefile.config   #edit-1

.SUFFIXES: 
.SUFFIXES: .cpp .dp.cpp .o  # edit-2

EXEC = SeisAcoMod2D_GPU_SYCL.V1.exe #edit

Objects = main.o json_parser.o modelling_worker.o modelling_master.o read_receiver.o create_geometry.o \
          workload.o model.o check_stability.o gpu_modelling_kernels.dp.o gpumodelling.dp.o # edit-3

all : $(EXEC)

$(EXEC):$(Objects)
	$(CPPLINKER) $(FFLAGS) -o $(EXEC) $(Objects) $(FLIBS) $(FFLAGS) #-L/home/apps/cuda-9.0/lib64/ -lcudart #edit-4
	mv $(EXEC) ../../../bin/  #edit-5

clean:
	rm -f *.o *~

#.cu.o:
#	nvcc $(NVFLAGS) -c $*.cu    #edit-6
.dp.cpp.o:
	$(CPPC) $(FFLAGS) -c $*.dp.cpp $(FLIBS) $(FFLAGS) #edit-7

.cpp.o:
	$(CPPC) $(FFLAGS) -c $*.cpp $(FLIBS) $(FFLAGS)

cleano:
	rm -f *.o *~	
