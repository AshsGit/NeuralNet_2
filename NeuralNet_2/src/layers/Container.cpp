#pragma once

#include "../internal/MatrixDispatcher.hpp"
#include "./Layer.hpp"
#include "./Container.hpp"

#include <vector>
#include <exception>

using std::vector;
using std::exception;


inline Layer::_LayerType Container::layer_type() { return Layer::_LayerType::Container };

/*
Note, this is a memory inefficient implementation but it shouldn't matter for this function.
*/
vector<Layer> Container::flatten_layer_vec(vector<Layer> layers) {
	vector<Layer> flat_layers;

	for (auto &layer : layers) {
		if (layer.layer_type() == Layer::_LayerType::Layer)
			flat_layers.push_back(layer);
		else {
			auto flat_sub_layers = flatten_layer_vec(((Container&)layer).layers);
			flat_layers.insert(flat_layers.end(), flat_sub_layers.begin(), flat_sub_layers.end());
		}
	}
	return flat_layers;
};


void Container::add_layers(vector<Layer> layers) { 
	auto& flat_layers = this->flatten_layer_vec(layers);
	this->_addLayersImp(flat_layers);

	vector<size_t> new_dims = this->_dimensions; // note, this copies the contents of this->_dimensions. It is not just a reference.

	if (new_dims.empty() && !layers.empty()) {
		new_dims.push_back(layers[0].dimensions()[0]);
	}
	size_t prev_layer_out = new_dims.back(); // TODO this is a terrible and unsafe way of making sure 

	for (auto& layer : flat_layers) {
		// Note, we can be sure layer.dimensions() is of size 2 as only containers can defy this and we have 
		// already flattened all containers via flatten_layer_vec (i.e. we can be sure layer is not a container).
		auto layer_in = layer.dimensions();
		auto layer_out = layer.dimensions(); 
		
		if (layer.in == prev_layer_out) {
			new_dims.push_back(layer_out);
			prev_layer_out = layer_out;
		}
		else {
			throw std::exception; // TODO: Create a better exception here when layers are incompatible within a container!
		}
	}

	this->_dimensions = new_dims;
};

