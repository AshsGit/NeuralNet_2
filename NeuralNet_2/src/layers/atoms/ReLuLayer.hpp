#pragma once

#include "../../internal/MatrixDispatcher.hpp"
#include "../../internal/types.hpp"

#include "../Layer.hpp"


class ReLuLayer : public Layer
{
public:
	ReLuLayer() = default;

	void forward_prop(MatrixDispatcher& dispatcher);
	void backward_prop(MatrixDispatcher& dispatcher);
};


