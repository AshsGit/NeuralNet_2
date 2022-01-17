#pragma once 

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::Vector;
using Eigen::RowVector;
using Eigen::Dynamic;

namespace types {
	using neuron_t = float;
	using Batch = Matrix<neuron_t, Dynamic, Dynamic>;

	using MatrixXneu = Matrix<neuron_t, Dynamic, Dynamic>;
	using VectorXneu = Vector<neuron_t, Dynamic>;
	using RowVectorXneu = Eigen::RowVector<neuron_t, Dynamic>;
};
