#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../Layer.hpp"

#include <vector>


using std::vector;


class Chain : public Layer
{
protected:
	vector<Layer> layers;

public:
	Chain(vector<Layer>& layers);

	void forward_prop(MatrixDispatcher& dispatcher);
	void backward_prop(MatrixDispatcher& dispatcher);
};


#include "./Chain.cpp"

