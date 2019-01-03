/*
 * neuralnet.cpp
 *
 *  Created on: Dec 27, 2018
 *      Author: oscarsplitfire
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cmath>

#include <fstream>
#include <sstream>


#define ETA 1.0 
#define ALPHA 0.5

typedef std::vector<double> VecVals;
typedef std::vector<int> Topo;

struct Connection {
	double weight, deltaWeight;
	Connection() {
		weight = rand() / double(RAND_MAX);
	}
};

// NEURON CLASS

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron {

public:
	Neuron(int numOutputs, int index);
	double outval;
	void neuronFeedFwd(const Layer &prevLayer);
	void calcOutGradients(double targetVal);
	void calcHiddenGradients(const Layer &next);
	void updateInWeights(Layer &prev);
private:
	static double transferFunc(double x);
	static double transferFuncDeriv(double x);
	double sumDOW(const Layer &next) const;
	std::vector<Connection> m_outWeights;
	int m_index;
	double m_gradient;
};

void Neuron::updateInWeights(Layer &prev) {
	for (int i = 0; i < prev.size(); i++) {
		Neuron &n = prev[i];
		double oldDelta = n.m_outWeights[m_index].deltaWeight;
		double newDelta =
				ETA
				* n.outval
				* m_gradient
				+ ALPHA
				* oldDelta;

		n.m_outWeights[m_index].deltaWeight = newDelta;
		n.m_outWeights[m_index].weight += newDelta;
	}
}

double Neuron::sumDOW(const Layer &next) const {
	double sum = 0.0;
	for (int i = 0; i < next.size() - 1; i++) {
		sum += (m_outWeights[i].weight * next[i].m_gradient);
	}
	return sum;
}

void Neuron::calcOutGradients(double targetVal) {
	double delta = targetVal - outval;
	m_gradient = delta * Neuron::transferFuncDeriv(outval);
}

void Neuron::calcHiddenGradients(const Layer &next) {
	double dow = sumDOW(next);
	m_gradient = dow * Neuron::transferFuncDeriv(outval);
}

Neuron::Neuron(int numOutputs, int index) {
	for (int i = 0; i < numOutputs; i++) {
		m_outWeights.push_back(Connection());
	}
	this->m_index = index;
}

double Neuron::transferFunc(double x) {
	return tanh(x);
}

double Neuron::transferFuncDeriv(double x) {
    return 1.0 - (x) * (x);
}

void Neuron::neuronFeedFwd(const Layer &prevLayer) {
	double sum = 0.0;
	for (Neuron el : prevLayer) {
		sum += el.outval * el.m_outWeights[m_index].weight;
	}
	outval = Neuron::transferFunc(sum);
}

// NEURON CLASS

// NEURAL NET CLASS

class NeuralNet {

public:
	NeuralNet(const Topo &topology);
	void feedFwd(const VecVals &inputVals);
	void backProp(const VecVals &targetVals);
	void getResults(VecVals &resultVals) const;
	inline double getrecentAvgErr() {return m_recentAvgErr;}

private:
	std::vector<Layer> m_layers; // [layerNum][neuronNum]
	double m_error;
	double m_recentAvgErr;
	double m_recentAvgSmoothingFactor;
};

void NeuralNet::getResults(VecVals &resultVals) const {
	resultVals.clear();
	for (int i = 0; i < m_layers.back().size(); i++) {
		resultVals.push_back(m_layers.back()[i].outval);
	}
}

NeuralNet::NeuralNet(const Topo &topology) {
	int numLayers = topology.size();
	// layer
	for (int i = 0; i < numLayers; i++) {
		m_layers.push_back(Layer());
		int numOutputs = (i == numLayers - 1) ? 0 : topology[i + 1];
		// neuron		  (add bias neuron)
		for (int j = 0; j <= topology[i]; j++) {
			m_layers[i].push_back(Neuron(numOutputs, j));
			printf("added Neuron layer %d	neuron %d\n", i, j);
		}
		m_layers.back().back().outval = 1.0;
	}
}

void NeuralNet::backProp(const VecVals &targetVals) {

	Layer &outLayer = this->m_layers.back();
	// calculate RMS
	m_error = 0.0;

	for (int i = 0; i < outLayer.size() - 1; i++) {
		double delta = targetVals[i] - outLayer[i].outval;
		m_error += delta * delta;
	}

	m_error = sqrt(m_error / (outLayer.size() - 1));

	m_recentAvgErr = (m_recentAvgErr * m_recentAvgSmoothingFactor + m_error)
			/ (m_recentAvgSmoothingFactor + 1.0);

	for (int i = 0; i < outLayer.size() - 1; i++) {
		outLayer[i].calcOutGradients(targetVals[i]);
	}

	for (int i = m_layers.size() - 2; i > 0; i--) {
		Layer &hidden = m_layers[i];
		Layer &next = m_layers[i + 1];
		for (int j = 0; j < hidden.size(); j++) {
			hidden[j].calcHiddenGradients(next);
		}
	}

	for (int i = m_layers.size() - 1; i > 0; i--) {
		Layer &layer = m_layers[i];
		Layer &prev = m_layers[i - 1];
		for (int j = 0; j < layer.size() - 1; j++) {
			layer[j].updateInWeights(prev);
		}
	}

}

void NeuralNet::feedFwd(const VecVals &inputVals) {

	if (inputVals.size() != m_layers[0].size() - 1) {
		fprintf(stderr,
		"Error: input values size (%lu) not equal to layer size - 1 (%lu)\n",
		inputVals.size(), m_layers[0].size() - 1);
		exit(1);
	}

	for (int i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].outval = inputVals[i];
	}

	// FORWARD PROPAGATION

	for (int i = 1; i < m_layers.size(); i++) {
		Layer &prevLayer = m_layers[i - 1];
		for (int j = 0; j < m_layers[i].size() - 1; j++) {
			m_layers[i][j].neuronFeedFwd(prevLayer);
		}
	}


}

// NEURAL NET CLASS

int main(void) {

	Topo topology;
	VecVals inputVals, targetVals, resultVals;

	std::ifstream fin;
	fin.open("trainme");

	std::stringstream ss;
	std::string str;

	std::getline(fin, str);
	ss.str(str);
	double d;
	int i;
	while (ss >> i) {
		topology.push_back(i);
	}
	ss.clear();

	NeuralNet net(topology);

	int pass = 0;

	while (fin.good()) {
		pass++;

        printf("pass:  %d\n", pass);

		inputVals.clear();
		targetVals.clear();

		std::getline(fin, str);
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

	return 0;
}




