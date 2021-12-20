include rules.mk

.PHONY: codegen.hip clean

build.hip: 
	$(HIPCC) $(HIPCC_CFLAGS) -c heat_kernel.cpp
	$(HIPFC) heat.F90 -c $(CFLAGS)
	$(HIPFC) -o heat heat.o heat_kernel.o $(CFLAGS)
clean:
	rm -rf heat *-gpufort.* *-fort2hip.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod log/
