#pragma once

#include "../internal/MatrixDispatcher.hpp"

#include <vector>
using std::vector;


class Layer
{
protected:
	enum _LayerType {Layer, Container};
	inline _LayerType layer_type() { return _LayerType::Layer };

	vector<size_t> _dimensions;

public:
	Layer(size_t in, size_t out) : _dimensions({in, out}) {};

	/*
	forward_prop should use the input vector from the provided dispatcher to genrate an 
	output vector which can be written to the dispatcher.

	Note: forward_prop should never alter the dispatcher input vector as it will be needed for back_prop.
	*/
	virtual void forward_prop(MatrixDispatcher& dispatcher) = 0;

	/*
	Let dC/dO := the derivative of the neural net cost by the output of the current layer.
	Let dC/dI := the derivative of the neural net cost by the input of the current layer (note, input of current layer = output of previous layer).

	The dispatcher input vector repressents dC/dO.
	The dispatcher output vector repressents the forward_prop input for the given layer. This will be overriden by this function and become dC/dI.

	This function uses dC/dO to perform all necessary gradient decent/backprop updates to the given layer.
	The function will then calculate dC/dI and write the result to the dispatcher output.

	Remember to use Eigen wisely when recieving input and writing output. Eigen uses lazy evaluation and thus it is usually sufficient to use the 
	pre-allocated dispatcher input and output vectors for all calculations to avoid unnecessary memory allocation and copying.
	*/
	virtual void backward_prop(MatrixDispatcher& dispatcher) = 0;

	inline const vector<size_t> dimensions();
};
