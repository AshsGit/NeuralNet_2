#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../../internal/types.hpp"

#include "../Layer.hpp"
#include "./Chain.hpp"

#include <vector>

using std::vector;


Chain::Chain(vector<Layer> layers) {
	for (auto layer : layers) {
		if (typeid(layer) == typeid(Chain)) { // TODO: Is type comparison slow in c++? if so find another solution/re-design.
			for (auto sublayer : static_cast<Chain*>(&layer)->layers)
				this->layers.push_back(sublayer);
		}
		else
			this->layers.push_back(layer);
	}
}


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

