#pragma once

#include "../internal/MatrixDispatcher.hpp"
#include "../internal/types.hpp"

#include "../Layer.hpp"


class BiasLayer : public Layer
{
	types::VectorXneu& _bias;

	// should be >0 for all numeric neuron_t types (otherwise becomes gradient accent which is not very useful).
	types::neuron_t _learning_rate; 

public:
	BiasLayer(types::VectorXneu& initBias) : _bias(initBias) {};

	void setLearningRate(types::neuron_t lr);

	void forward_prop(MatrixDispatcher& dispatcher);
	void backward_prop(MatrixDispatcher& dispatcher);
};


#include "./BiasLayer.cpp"
