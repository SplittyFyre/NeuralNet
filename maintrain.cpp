/*
 * main.cpp
 *
 *  Created on: Dec 27, 2018
 *      Author: oscarsplitfire
 */

#include "libneuralnet/neuralnet.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>

#include <fstream>
#include <sstream>

int main(int argc, char *argv[]) {

	/*FILE *nin = fopen("test", "r");
	NeuralNet mynet{};
	NeuralNet::readNeuralNet(nin, mynet);

	FILE *nout = fopen("check", "w");
	NeuralNet::printNeuralNet(nout, mynet);

	VecVals input;
	input.push_back(1);
	input.push_back(1);
	mynet.feedFwd(input);

	VecVals result;
	mynet.getResults(result);

	for (double el : result) {
		printf("%f\n", el);
	}

	exit(0);*/

	if (argc != 3) {
		std::cerr << "error: args are    training file    outputnetfile\n";
		return 1;
	}

	Topo topology;
	VecVals inputVals, targetVals, resultVals;

	std::ifstream fin;
	fin.open(argv[1]);

	double e, a;

	fin >> e >> a;

	printf("eta: %f, alpha: %f\n", e, a);

	std::stringstream ss;
	std::string str;
	std::getline(fin, str);

	std::getline(fin, str);
	ss.str(str);
	double d;
	int i;
	while (ss >> i) {
		topology.push_back(i);
	}
	ss.clear();

	NeuralNet net(topology);
	net.setEta(e);
	net.setAlpha(a);

	int pass = 0;

	while (fin.good()) {
		pass++;

        printf("training pass:  %d\n", pass);

		inputVals.clear();
		targetVals.clear();

		std::getline(fin, str);
		if (str.empty()) {
			puts("line empty, breaking");
			break;
		}
		ss.str(str);
		while (ss >> d) {
			inputVals.push_back(d);
		}
		ss.clear();

		net.feedFwd(inputVals);
		printf("Inputs:  ");
		for (double el : inputVals) {
			printf("%f , ", el);
		}
		puts("");


		net.getResults(resultVals);
		printf("Outputs:  ");
		for (int i = 0; i < resultVals.size() - 1; i++) {
			printf("%f , ", resultVals[i]);
		}
		puts("");


		std::getline(fin, str);
		if (str.empty()) {
			puts("line empty, breaking");
			break;
		}
		ss.str(str);
		while (ss >> d) {
			targetVals.push_back(d);
		}
		ss.clear();

		net.backProp(targetVals);
		printf("Targets:  ");
		for (double el : targetVals) {
			printf("%f , ", el);
		}
		puts("");

		printf("Net recent average error:  %f\n", net.getrecentAvgErr());

        puts("");

	}

	FILE *fout = fopen(argv[2], "w");
	NeuralNet::printNeuralNet(fout, net);

	return 0;
}



