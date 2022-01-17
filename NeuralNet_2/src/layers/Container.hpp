#pragma once

#include "../internal/MatrixDispatcher.hpp"
#include "./Layer.hpp"

#include <vector>

using std::vector;


class Container : public Layer
{
private:
	vector<Layer> flatten_layer_vec(vector<Layer> layers);

protected:
	vector<Layer> layers; // This should contain a list of Layer's (NOT Containers). The containers
	
	inline _LayerType layer_type() { return _LayerType::Container };

	virtual void _add_layers_imp(vector<Layer> layers) = 0;
	
public:
	Container(size_t in, size_t out) = delete;

	void add_layers(vector<Layer> layers);
	virtual void forward_prop(MatrixDispatcher& dispatcher) = 0;
	virtual void backward_prop(MatrixDispatcher& dispatcher) = 0;
};
