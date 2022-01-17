#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../../internal/types.hpp"

#include "../Layer.hpp"
#include "./ReLuLayer.hpp"


inline auto relu(types::neuron_t n) { return n > 0 ? n : 0; }
inline auto relu_deriv(types::neuron_t n) { return n > 0 ? 1 : 0; }


void ReLuLayer::forward_prop(MatrixDispatcher& disp)
{
	disp.output() = disp.input().unaryExpr(&relu);
};


/*
Note, ReLu has no configurable weights, thus no updates are performed here. 
This function simply calculates dC/dI and writes its transpose back to the dispatcher.
*/
void ReLuLayer::backward_prop(MatrixDispatcher& disp)
{
	auto dC_by_dO = disp.input();
	auto forward_prop_layer_input = disp.output();

	auto dC_by_dI = dC_by_dO.array() * forward_prop_layer_output.unaryExpr(&relu_deriv).array();
	
	disp.output() = dC_by_dI;
};

