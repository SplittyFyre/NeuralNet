/*
 * neuralnet.h
 *
 *  Created on: Dec 27, 2018
 *      Author: oscarsplitfire
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <vector>

typedef std::vector<double> VecVals;
typedef std::vector<int> Topo;

struct Connection {
	double weight, deltaWeight;
	Connection() {
		weight = rand() / double(RAND_MAX);
	}
};

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron {

public:
	Neuron(int numOutputs, int index);
	double outval;
	void neuronFeedFwd(const Layer &prevLayer);
	void calcOutGradients(double targetVal);
	void calcHiddenGradients(const Layer &next);
	void updateInWeights(Layer &prev, double eta, double alpha);

	static void printNeuron(FILE *stream, const Neuron &n);
	static void readNeuron(FILE *stream, Neuron &n);

private:
	static double transferFunc(double x);
	static double transferFuncDeriv(double x);
	double sumDOW(const Layer &next) const;
	std::vector<Connection> m_outWeights;
	int m_index;
	double m_gradient;
};

class NeuralNet {

public:
	NeuralNet(const Topo &topology);
	NeuralNet();
	void feedFwd(const VecVals &inputVals);
	void backProp(const VecVals &targetVals);
	void getResults(VecVals &resultVals) const;
	inline double getrecentAvgErr() {return m_recentAvgErr;}

	static void printNeuralNet(FILE *stream, const NeuralNet &nn);
	static void readNeuralNet(FILE *stream, NeuralNet &nn);

	inline double getEta() {return m_eta;}
	inline double getAlpha() {return m_alpha;}

	inline void setEta(double eta) {m_eta = eta;}
	inline void setAlpha(double alpha) {m_alpha = alpha;}

private:
	std::vector<Layer> m_layers; // [layerNum][neuronNum]
	double m_error;
	double m_recentAvgErr;
	double m_recentAvgSmoothingFactor;
	double m_eta = 1.0;
	double m_alpha = 0.6;
};

#endif /* NEURALNET_H_ */
