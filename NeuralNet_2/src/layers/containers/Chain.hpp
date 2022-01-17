#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../Container.hpp"

#include <vector>


using std::vector;


class Chain : public Container
{
protected:
	void _add_layers_imp(vector<Layer> layers);

public:
	Chain(vector<Layer> layers);

	/*
	This method forward propogates through all sub-layers of the chain.
	*/
	void forward_prop(MatrixDispatcher& dispatcher);

	/*
	This method back propogates through all sub-layers of the chain.
	*/
	void backward_prop(MatrixDispatcher& dispatcher);
};


#include "./Chain.cpp"

