#include "../NeuralNet_2/src/ANNExecutor.hpp"
#include "../NeuralNet_2/src/layers/FeedForwardLayer.hpp"
#include "../NeuralNet_2/src/layers/containers/Chain.hpp"

#include "../NeuralNet_2/src/internal/types.hpp"

#include <iostream>
#include <vector>

using namespace std;

using types::Batch;


int main() {
	// XOR example

	Batch input{
		{1, 1, 0, 0},
		{1, 0, 1, 0},
	};
	types::RowVectorXneu expectedOutput{ {0, 1, 1, 0} };

	auto ann = new ANNExecutor(new Chain({
		new FeedForwardLayer(2, 2, 2),
		new FeedForwardLayer(2, 1, 2)
		}));

	for (int i = 0; i < 100; i++) { // 100 epochs since it is simple XOR
		auto& disp = ann->forward(input);
		ann->backward(disp);
	};

	cout << ann->run(input) << endl;
	return 0;
};