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

    if (argc != 3) {
        std::cerr << "error: args are    testfile    inputnetfile\n";
        return 1;
    }

	FILE *nin = fopen(argv[2], "r");
	NeuralNet net{};
	NeuralNet::readNeuralNet(nin, net);

	VecVals inputVals, targetVals, resultVals;

	std::ifstream fin;
	fin.open(argv[1]);

    double d;

	std::stringstream ss;
	std::string str;

	int pass = 0;

	while (fin.good()) {
		pass++;

        printf("test pass:  %d\n", pass);

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

		printf("Targets:  ");
		for (double el : targetVals) {
			printf("%f , ", el);
		}
		puts("");

		printf("Net recent average error:  %f\n", net.getrecentAvgErr());

        puts("");

	}

	return 0;
}



