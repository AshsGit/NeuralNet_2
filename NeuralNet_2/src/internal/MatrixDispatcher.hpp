#pragma once


#include <Eigen/Dense>

#include "./types.hpp"

/*
Cannot make this static size as this would cause too many issues considering the network is constructed at runtime
E.g. Program would need to know the exact number of layers (and their sizes) at compile time. This would include any layers nested within a a Chain (which semi defeats the purpose of a chain).

Consequences:
* ANN cannot be statically sized
*/
class MatrixDispatcher
{
public:
	types::MatrixXneu& input();
	types::MatrixXneu& output();

	void nextLayer();
	void reverse();

	int batchSize();
};