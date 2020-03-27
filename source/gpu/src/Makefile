OBJS=GateGPUManager.ptx

NVCC=nvcc
NVCCFLAGS=

.PHONY: all
all: $(OBJS)

.PHONY: clean
clean:
	$(RM) $(OBJS)


%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx --include-path ../../digits_hits/include/ -I /home/agmez/gate/geant4.10.05.p01/source/run/include/ -I /home/agmez/gate/geant4.10.05.p01/source/global/management/include/ -I /home/agmez/gate/geant4.10.05.p01/source/event/include/ -I /home/agmez/gate/geant4.10.05.p01/source/particles/management/include/ -I /home/agmez/gate/geant4.10.05.p01/source/externals/clhep/include/ -I /home/agmez/gate/geant4.10.05.p01/install/include/Geant4/ -I /home/agmez/gate/Gate-Julia/build/ -I /home/agmez/gate/Gate-Julia/source/general/include/ -I /home/agmez/gate/root/io/io/inc/ -I /home/agmez/gate/root/core/zip/inc/ -I /home/agmez/gate/root/core/base/inc/ -I /home/agmez/gate/root/install/include/ -I /home/agmez/gate/Gate-Julia/source/geometry/include/ -I /home/agmez/gate/Gate-Julia/source/externals/itk-mhd/src/ -I /home/agmez/gate/Gate-Julia/source/externals/itk-mhd/ -I /home/agmez/gate/Gate-Julia/source/externals/itk-mhd/src/ -I /home/agmez/gate/Gate-Julia/build/itk-mhd/ -I /home/agmez/gate/Gate-Julia/source/physics/include/ -I/home/agmez/gate/Gate-Julia/source/gpu/include/ $^ -o $@
