#pragma once

#include "ANNExecutor.hpp"

#include "internal/types.hpp"
#include "./Layer.hpp"
#include "./internal/MatrixDispatcher.hpp"

#include <vector>
using std::vector;


ANNExecutor::ANNExecutor(Layer layer) {
	this->layer = layer;
};


types::MatrixXneu& ANNExecutor::run(types::Batch input) {
	this->forward(input).output(); // TODO: check if need to call nextLayer or go to previous layer.
};
	

MatrixDispatcher& ANNExecutor::forward(types::Batch input);


void ANNExecutor::backward(MatrixDispatcher& disp);
	
