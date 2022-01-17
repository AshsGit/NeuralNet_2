#pragma once

#include "internal/types.hpp"
#include "./Layer.hpp"
#include "./internal/MatrixDispatcher.hpp"

#include <vector>
using std::vector;


class ANNExecutor
{
protected:
	
	Layer network;
	void* cost_func; // TODO

public:
	ANNExecutor(Layer network);

	types::MatrixXneu& run(types::Batch input);
	
	MatrixDispatcher& forward(types::Batch input);
	void backward(MatrixDispatcher& disp);
	
};