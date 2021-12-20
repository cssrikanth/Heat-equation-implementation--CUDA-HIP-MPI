include rules.mk

.PHONY: build.hip codegen.hip clean

codegen.hip:
	#gpufort -w heat.F90 -E hip --config-file options.py.in --log-level DEBUG --emit-cpu-impl --emit-debug-code
	gpufort -w heat.F90 -E hip --config-file options.py.in --log-level DEBUG --emit-cpu-impl 
build.hip:
	$(HIPCC) $(HIPCC_CFLAGS) -c mod_heat-fort2hip.hip.cpp
	$(HIPCC) $(HIPCC_CFLAGS) -c mod_swap-fort2hip.hip.cpp
	$(HIPFC) heat.F90-gpufort.f08 -c $(CFLAGS)
	$(HIPFC) -o heat mod_heat-fort2hip.hip.o mod_swap-fort2hip.hip.o heat.F90-gpufort.o $(CFLAGS)
clean:
	rm -rf heat *-gpufort.* *-fort2hip.* *.o *.mod gpufort*.h $(TEST_NAME) *.gpufort_mod log/
