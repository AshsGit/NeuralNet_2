#pragma once

#include "./Layer.hpp"
#include "./internal/MatrixDispatcher.hpp"

#include <vector>

class Batch; // this is a placeholder type

template<class _T>
class ANN
{
protected:
	
	vector<Layer> layers;
	void* cost_func; // TODO

public:
	int in(); //return the number of inputs to the ANN (of type _T)
	int out(); //return the number of outputs to the ANN (of type _T)

	vector<Layer>& getLayers();

	MatrixDispatcher forward_prop(Batch input);
	void backward_prop(MatrixDispatcher input);
	
};