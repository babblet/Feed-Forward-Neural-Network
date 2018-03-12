#ifndef NN_CLASS_H
#define NN_CLASS_H
#include <vector>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <iostream>

// NNClass:
//  Feedforward Neural Network with dynamic sizing.
//  Able to hande dynamic size of input. 
//  Multiple inputs needs to be in same size.
//  Can only handle input in vector form.
//  Free sizing of hidden layers and output.
class NNClass
{
	public:
		NNClass(int depth, float constant, int input_size, std::vector<int> layer_size); //Create a Neural Network with specified size

		// load(string filepath);         // Load Neural Network
		// save(string path, string filename);      // Save Neural Network
		void train(std::vector<std::vector<float> > input, std::vector<std::vector<float> > target, int interations);   // Train Feedforward Neural Network with input as an float array
		//bool destroy();

	private:
		//Layer Data, Keeps the information of a single layer.
		//A layer is with the weights as input to a neuron and its output.
		//Also keeps the information of the delta value for the neurons in the layer.
		struct Layer_struct
		{
			std::vector<float> theta;
			std::vector<float> delta;
			std::vector<float> output;
			std::vector<std::vector<float> > weight;
		};

		std::vector<Layer_struct> layer;
		std::vector<int> layer_size; // Holds the number of neurons in each layer

		float constant;   // Learning rate
		int depth;    // Depth of current Neural Network
		int input_size;   // Holds the size of the given input

		bool allocate_layers();   // Allocates the layer struct
		bool randomize_weigths(); // Randomizes weights
		bool backpropagation(std::vector<float> input, std::vector<float> target);    // Backpropagation algorithm taken from Lars Asplund 
		float activation(int layer, int index, std::vector<float> input);    // Needs to be specified.
};
#endif
