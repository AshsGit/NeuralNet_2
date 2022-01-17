#pragma once

#include "MatrixDispatcher.hpp"

#include "../layers/Layer.hpp"
#include "./types.hpp"

#include <Eigen/Dense>



MatrixDispatcher::MatrixDispatcher(Layer& layer, size_t batch_size) {
	for (auto dim : layer.dimensions()) {
		this->_matrices.push_back(new types::MatrixXneu(dim, batch_size));
	}
};

types::MatrixXneu& MatrixDispatcher::input() {
	return *(this->_iterator);
};

types::MatrixXneu& MatrixDispatcher::output() {
	if (this->_reversed)
		return *(this->_iterator - 1);
	else
		return *(this->_iterator + 1);
};

void MatrixDispatcher::nextLayer() {
	if (this->_reversed)
		this->_iterator -= 1;
	else
		this->_iterator += 1;
};

void MatrixDispatcher::reverse() {
	this->_reversed = !this->_reversed;
};

int MatrixDispatcher::batchSize() {
	return this->_batch_size;
};
