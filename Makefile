EXEC      := nvcc
NVCCFLAGS := --gpu-architecture=sm_50 --use_fast_math  -g -G -Xcompiler -rdynamic -lineinfo
INCLUDES  := -I./common -I./common/UtilNPP
LIBS      := -L/usr/local/cuda-11.8/targets/x86_64-linux/lib/ -lnppif -lnppist -lnppisu -lnppc -lculibos -lnppicc -lfreeimage

clean:
	rm -f bin/* output/* 2>/dev/null | true

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

main.o:
	$(EXEC) $(NVCCFLAGS) $(INCLUDES) $(LIBS) -o bin/main.o -c src/main.cpp

main: main.o
	$(EXEC) $(NVCCFLAGS) $(INCLUDES) $(LIBS) -o bin/main bin/main.o

build: main

run:
	./bin/main $$(ls data/*)
# run:
# 	./bin/main data/image11.jpg

all: clean build run