#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../../internal/types.hpp"

#include "../Layer.hpp"
#include "./FCLayer.hpp"


void FCLayer::forward_prop(MatrixDispatcher& disp)
{
	disp.output() = this->_weights * disp.input();
};


void FCLayer::backward_prop(MatrixDispatcher& disp)
{
	auto dC_by_dO = disp.input();
	auto in = disp.output(); // layer input from forward propogation 
	
	disp.output() = dC_by_dO * this->_weights; // dC/dI = dC/dO * w;

	//weight updates
	auto dC_by_dw_T = in * dC_by_dO / disp.batchSize(); // TODO: is it necessary 

	this->_weight -= this->_learning_rate * dC_by_dw_T.transpose();	
};

