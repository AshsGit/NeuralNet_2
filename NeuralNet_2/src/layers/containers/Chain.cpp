#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../../internal/types.hpp"

#include "../Layer.hpp"
#include "./Chain.hpp"

#include <vector>

using std::vector;


void Chain::_add_layers_imp(vector<Layer> layers) {
	this->layers.insert(this->layers.end(), layers.begin(), layers.end());
};


Chain::Chain(vector<Layer> layers) {
	this->add_layers(layers);
};


void Chain::forward_prop(MatrixDispatcher& disp)
{
	for (int i; i < this->layers.size() - 1 ; i++) {
		this->layers[i].forward_prop(disp);
		disp.nextLayer();
	}

	// While this is ugly, the final iteration of the above loop must be executed outside the loop 
	// so that disp.nextLayer() is not run in the last loop iteration.
	this->layers[this->layers.size() - 1].forward_prop(disp);

};


void Chain::backward_prop(MatrixDispatcher& disp)
{
	for (int i; i < this->layers.size() - 1; i++) {
		this->layers[i].backward_prop(disp);
		disp.nextLayer();
	}

	// While this is ugly, the final iteration of the above loop must be executed outside the loop 
	// so that disp.nextLayer() is not run in the last loop iteration.
	this->layers[this->layers.size() - 1].forward_prop(disp);
};

