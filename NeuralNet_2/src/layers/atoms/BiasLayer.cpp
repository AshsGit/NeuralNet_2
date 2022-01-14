#pragma once

#include "../internal/MatrixDispatcher.hpp"
#include "../../internal/types.hpp"

#include "./BiasLayer.hpp"



void BiasLayer::setLearningRate(types::neuron_t lr)
{
	this->_learning_rate = lr; // scalling cannot happen until batch size is known so cant reduce scalling overhead :(
}


void BiasLayer::forward_prop(MatrixDispatcher& disp) 
{
	disp.output() = disp.input() + this->_bias.rowwise().replicate(disp.batchSize());
};


void BiasLayer::backward_prop(MatrixDispatcher& dispatcher)
{
	auto dC_by_dO = dispatcher.input();
	
	dispatcher.output() = dC_by_dO; // dC_by_dI = dC_by_dO in this case

	//bias updates
	auto dC_by_db = dC_by_dO.colwise().sum() / dispatcher.batchSize();

	this->_bias -= this->_learning_rate * dC_by_db;

	
};

