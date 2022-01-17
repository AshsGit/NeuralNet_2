#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../../internal/types.hpp"

#include "../Layer.hpp"


/*
Basic fully connected weighted layer.
*/
class FCLayer : public Layer
{
	types::MatrixXneu& _weights;

	// should be >0 for all numeric neuron_t types (otherwise becomes gradient accent which is not very useful).
	types::neuron_t _learning_rate;

public:
	FCLayer(types::MatrixXneu& initWeights, types::neuron_t learning_rate) : 
		_weights(initWeights), _learning_rate(learning_rate) {};

	void forward_prop(MatrixDispatcher& dispatcher);
	void backward_prop(MatrixDispatcher& dispatcher);
};


