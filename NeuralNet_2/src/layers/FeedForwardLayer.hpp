#pragma once

#include "../internal/MatrixDispatcher.hpp"
#include "../internal/types.hpp"

#include "./containers/Chain.hpp"


class FeedForwardLayer: public Chain
{

public:
	FeedForwardLayer(size_t in, size_t out, types::neuron_t learning_rate);

};


