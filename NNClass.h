#ifndef NN_CLASS_H
#define NN_CLASS_H
#include <vector>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

//NNClass:
//  Feedforward Neural Network with dynamic sizing.
//  Able to hande dynamic size of input. 
//  Multiple inputs needs to be in same size.
//  Can only handle input in vector form.
//  Free sizing of hidden layers and output.
class NNClass
{
	public:
		NNClass(int data_size, std::vector<int> &layer_size); //Create a Neural Network with specified size
		NNClass(int data_size, std::string &filename);
		void train(std::vector<std::vector<float> > &input, std::vector<std::vector<float> > &target, float constant, int epochs);
		void save(std::string filename);      // Save Neural Network
		void destroy(){delete this;};
	private:
		//Layer Data, Keeps the information of a single layer.
		//A layer is with the weights as input to a neuron and its output.
		//Also keeps the information of the delta value for the neurons in the layer.
		struct Layer_struct
		{
			std::vector<float> delta;
			std::vector<std::vector<float> > output;
			std::vector<std::vector<float> > weight;
		};

		std::vector<int> layer_size; // Holds the number of neurons in each layer
		std::vector<Layer_struct> layer;
		std::vector<std::vector<float> > target;		

		float constant; // Learning rate
		int depth;    	// Depth of current Neural Network
		int input_size; // Size of the given input
		int data_size;
		int epoch;
	
		//Setup functions
		bool allocate_layers();   // Allocates the layer struct
		bool randomize_weights(); // Randomizes weights

		//Alghoritms
		float cost();
		bool backpropagation(int data);   
		float activation(int layer, int index, int data);    // Needs to be specified.
};
#endif
