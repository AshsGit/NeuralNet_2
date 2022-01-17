#pragma once

#include "../internal/MatrixDispatcher.hpp"
#include "../internal/types.hpp"

#include "./FeedForwardNetwork.hpp"

#include "./atoms/BiasLayer.hpp"
#include "./atoms/FCLayer.hpp"
#include "./atoms/ReLuLayer.hpp"


FeedForwardLayer::FeedForwardLayer(size_t in, size_t out, types::neuron_t learning_rate) {
	this->add_layers({
			new FCLayer(types::MatrixXneu::Random(in, out), learning_rate),
			new BiasLayer(types::VectorXneu::Random(out), learning_rate),
			new ReLuLayer()
		});
};
