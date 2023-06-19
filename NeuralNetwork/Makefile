ARMA_FLAGS = -larmadillo -llapack -lblas
complie:
	g++ */*/*.cpp */*.cpp *.cpp -o mlp.exe $(ARMA_FLAGS)
debug:
	g++ */*/*.cpp */*.cpp *.cpp -o debug.exe -g -DVERBOSE -DINFO -DDEBUG $(ARMA_FLAGS)
clean:
	rm mlp.exe