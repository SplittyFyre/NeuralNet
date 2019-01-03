/*
 * neuralnet.cpp
 *
 *  Created on: Dec 27, 2018
 *      Author: oscarsplitfire
 */

#include "neuralnet.h"
#include <cmath>

// NEURON CLASS

void Neuron::updateInWeights(Layer &prev, double eta, double alpha) {
	for (int i = 0; i < prev.size(); i++) {
		Neuron &n = prev[i];
		double oldDelta = n.m_outWeights[m_index].deltaWeight;
		double newDelta =
				eta
				* n.outval
				* m_gradient
				+ alpha
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

NeuralNet::NeuralNet() {

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
			layer[j].updateInWeights(prev, m_eta, m_alpha);
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

void Neuron::printNeuron(FILE *stream, const Neuron &n) {
	fprintf(stream, "\n%.30f %.30f\n", n.m_gradient, n.outval);
	for (Connection const &c : n.m_outWeights) {
		fprintf(stream, "\n    %.30f %.30f\n", c.weight, c.deltaWeight);
	}
}


void Neuron::readNeuron(FILE *stream, Neuron &n) {
	fscanf(stream, "%lf %lf", &n.m_gradient, &n.outval);
	for (Connection &c : n.m_outWeights) {
		fscanf(stream, "    %lf %lf", &c.weight, &c.deltaWeight);
	}
}




void NeuralNet::printNeuralNet(FILE *stream, const NeuralNet &nn) {

	fprintf(stream, "%.3f %.3f\n", nn.m_eta, nn.m_alpha);
	fprintf(stream, "%.30f %.30f %.30f\n", nn.m_error, nn.m_recentAvgErr, nn.m_recentAvgSmoothingFactor);

	unsigned long numLayers = nn.m_layers.size();
	fprintf(stream, "%lu", numLayers);
	for (int i = 0; i < numLayers; i++) {
		fprintf(stream, " %lu", nn.m_layers[i].size() - 1);
	}
	fputs("\n", stream);

	for (int i = 0; i < numLayers; i++) {
		for (int j = 0; j < nn.m_layers[i].size(); j++) {
			Neuron::printNeuron(stream, nn.m_layers[i][j]);
		}
	}

}



void NeuralNet::readNeuralNet(FILE *stream, NeuralNet &nn) {

	fscanf(stream, "%lf %lf", &nn.m_eta, &nn.m_alpha);
	fscanf(stream, "%lf %lf %lf", &nn.m_error, &nn.m_recentAvgErr, &nn.m_recentAvgSmoothingFactor);

	unsigned long numLayers;
	fscanf(stream, "%lu", &numLayers);

	Topo topology;

	for (int i = 0; i < numLayers; i++) {
		int tmp;
		fscanf(stream, "%d", &tmp);
		topology.push_back(tmp);
	}

	for (int i = 0; i < numLayers; i++) {
		nn.m_layers.push_back(Layer());
		int numOutputs = (i == numLayers - 1) ? 0 : (topology[i + 1]);
		for (int j = 0; j <= topology[i]; j++) {
			Neuron neuron(numOutputs, j);
			Neuron::readNeuron(stream, neuron);
			nn.m_layers[i].push_back(neuron);
			printf("read Neuron layer %d	neuron %d\n", i, j);
		}
		nn.m_layers.back().back().outval = 1.0;
	}

}
