train:
	g++ -std=c++11 maintrain.cpp libneuralnet/neuralnet.cpp libneuralnet/neuralnet.h
	mv a.out nnbin/nettrain
test:
	g++ -std=c++11 maintest.cpp libneuralnet/neuralnet.cpp libneuralnet/neuralnet.h
	mv a.out nnbin/nettest
all:
	make train
	make test
