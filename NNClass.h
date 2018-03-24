#ifndef NN_CLASS_H
#define NN_CLASS_H
#include <vector>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

class NNClass
{
	public:		//  data_size = amount of training data
		NNClass(int data_size, std::vector<int> &layer_size); //Create new network with specified size.
		NNClass(int data_size, std::string &filename); // Load weights from file
		void train(std::vector<std::vector<float> > &input, std::vector<std::vector<float> > &target, float constant, int epochs);
		void save(std::string filename);      // Save weights
		void destroy(){delete this;};
	private:
		//Layer data
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
		int data_size;
	
		//Setup functions
		bool allocate_layers();
		bool randomize_weights();

		//Alghoritms
		float cost();
		bool backpropagation(int data);   
		float activation(int layer, int index, int data);
};
#endif
