#include <cmath>
#include <time.h>
#include <stdlib.h>
#include "iostream"
#include "NNClass.h"

#define DEPTH 3
#define PARAM_SIZE 2
#define CONSTANT 0.005
#define PI 3.1415926535
#define DATA_SIZE 3000

bool write_to_serial(float *input, int input_size)
{
	for(int index = 0; index < input_size; input++)
	{
		std::cout << input[index] << std::endl;
	}
}

//Used in randomize_weigths();
float random_weight(){
	//Return random float between -1 and 1
	srand(time(NULL));
	return ((float)rand() / ((float)RAND_MAX / 2)) - 1; 
}

//** Notes ** 
//  Implement error checking
//  Decide on public and private variables
//  Find better method for rand?
//  Make train able to hande several iterations with an array of inputs
//
// NNClass:
//  Feedforward Neural Network with dynamic sizing.
//  Able to hande dynamic size of input. 
//  Multiple inputs needs to be in same size.
//  Can only handle input in vector form.
//  Free sizing of hidden layers and output.

class NNClass
{
	public:
		NNClass(int depth, float constant, int input_size, int * layer_size); //Create a Neural Network with specified size

		// load(string filepath);         // Load Neural Network
		// save(string path, string filename);      // Save Neural Network
		void train(float ** input, float ** target, int interations);   // Train Feedforward Neural Network with input as an float array
		bool destroy();

	private:
		//Layer Data, Keeps the information of a single layer.
		//A layer is with the weights as input to a neuron and its output.
		//Also keeps the information of the delta value for the neurons in the layer.
		struct Layer_struct
		{
			float *  theta;
			float *  delta;
			float *  output;
			float ** weight;
		} * layer;

		float constant;   // Learning rate
		int depth;    // Depth of current Neural Network
		int input_size;   // Holds the size of the given input
		int * layer_size; // Holds the number of neurons in each layer

		bool allocate_layers();   // Allocates the layer struct
		bool randomize_weigths(); // Randomizes weights
		bool backpropagation(float * input, float * target);    // Backpropagation algorithm taken from Lars Asplund 
		float activation(int layer, int index, float * input);    // Needs to be specified.
};

NNClass::NNClass(int depth, float constant, int input_size, int * layer_size)
{
	this->depth = depth;
	this->constant = constant;
	this->input_size = input_size;
	this->layer_size = layer_size;
	this->allocate_layers();
	this->randomize_weigths();
}

//NNClass::load(string filepath){};
//NNClass::save(string path, string filename){};

//Singe activation (for now)
void NNClass::train(float ** input,  float ** target, int iterations)
{
	float * layer_input;
	for(int data = 0; data < iterations; data++)
	{
		for(int layer = 0; layer < this->depth; layer++)
		{
			if(layer > 0) layer_input = this->layer[layer - 1].output;
			else          layer_input = input[data];

			for(int index = 0; index < this->layer_size[layer]; index++)
			{
				this->layer[layer].output[index] = this->activation(layer, index, layer_input);
				this->backpropagation(input[data], target[data]);
			}
		}
	}
	write_to_serial(this->layer[this->depth - 1].output, PARAM_SIZE);
}

bool NNClass::allocate_layers()
{
	//Allocate layer pointer array of set depth
	this->layer = (NNClass::Layer_struct *) calloc(this->depth, sizeof(this->layer));

	//Allocate layers.
	for(int layer = 0; layer < this->depth; layer++)
	{
		int amount_weight_groups;
		if(layer > 0) amount_weight_groups = this->layer_size[layer - 1]; // Rest of layers
		else          amount_weight_groups = this->input_size;            // Input layer

		//Weight groups
		this->layer[layer].weight = (float **) calloc(amount_weight_groups, sizeof(float *));

		//Weights
		for(int weight_group = 0; weight_group < amount_weight_groups; weight_group++)
			this->layer[layer].weight[weight_group] = (float *) calloc(this->layer_size[layer], sizeof(float));

		//Neurons
		this->layer[layer].theta  = (float *) calloc(this->layer_size[layer], sizeof(float));
		this->layer[layer].delta  = (float *) calloc(this->layer_size[layer], sizeof(float));
		this->layer[layer].output = (float *) calloc(this->layer_size[layer], sizeof(float));
	}
	return true;
}

bool NNClass::destroy()
{
	for(int layer = 0; layer < this->depth; layer++)
	{
		int amount_weight_group;
		if(layer > 0) amount_weight_group = this->layer_size[layer - 1];
		else          amount_weight_group = this->input_size;

		for(int weight_group = 0; weight_group < amount_weight_group; weight_group++)
		{
			free(this->layer[layer].weight[weight_group]);
		}
		free(this->layer[layer].theta);
		free(this->layer[layer].delta);
		free(this->layer[layer].output);
	}
	return true;
}

bool NNClass::randomize_weigths()
{

	//Set random weigths
	for(int layer = 0; layer < this->depth; layer++)
	{
		//Setup
		int amount_weight_group;
		if(layer > 0) amount_weight_group = this->layer_size[layer - 1]; // Rest of layers
		else          amount_weight_group = this->input_size;              // Input layer

		//Set
		for(int weight_group = 0; weight_group < amount_weight_group; weight_group++)
		{
			for(int weight = 0; weight < this->layer_size[layer]; weight++)
			{
				this->layer[0].weight[weight_group][weight] = random_weight();
			}
		}
	}
	return true;
}

//Com
bool NNClass::backpropagation(float * input, float * target)
{
	//Setup pointers for easier reading
	float *  theta,
	      *  delta,
	      *  delta_last,
	      *  output,
	      *  constant,
	      *  layer_input,
	      ** weight;

	//Target delta update
	//
	//Setup
	output = this->layer[this->depth - 1].output;
	delta  = this->layer[this->depth - 1].delta;

	//Update
	for(int index = 0; index < layer_size[this->depth - 1]; index++)
		delta[index] = (target[index] - output[index]) * output[index] * (1 - output[index]);

	//Hidden delta update
	for(int layer = this->depth - 2; layer >= 0; layer--)
	{
		//Setup
		theta  = this->layer[layer].theta;
		delta  = this->layer[layer].delta;
		weight = this->layer[layer].weight;
		delta_last = this->layer[layer - 1].delta;

		//Update
		for(int index = 0; index < this->layer_size[layer]; index++)
		{
			float sum = 0;

			//Calculate sum of the previous deltas and weights
			for(int weight_index = 0; weight_index < this->layer_size[layer + 1]; weight_index++)
				sum += delta_last[weight_index] * weight[index][weight_index];

			//Calculate delta for current layer and group
			delta[index] = output[index] * (1 - output[index]) * sum;
		}
	}

	//Weight and Theta update
	for(int layer = 0; layer < this->depth; layer++)
	{
		//Setup
		weight   = this->layer[layer].weight;
		theta    = this->layer[layer].theta;
		delta    = this->layer[layer].delta;

		int amount_weight_group;
		if(layer > 0){
			amount_weight_group = this->layer_size[layer - 1];
			layer_input = this->layer[layer - 1].output;
		}
		else{
			amount_weight_group = this->input_size;
			layer_input = input;
		}

		//Update
		for(int index = 0; index < this->layer_size[layer]; index++)
		{
			for(int weight_group = 0; weight_group < amount_weight_group; weight_group++)
				weight[weight_group][index] += (this->constant * delta[index] * layer_input[weight_group]);

			theta[index] += (this->constant * delta[index]);
		}
	}
	return true;
}

//Round activation | 0.99 -> 1 | 0.01 -> 0 |
float NNClass::activation(int layer, int index, float * input)
{
	//Setup
	float sum = 0;
	int amount_weight_group;

	if(layer > 0) amount_weight_group = this->layer_size[layer - 1];
	else          amount_weight_group = this->input_size;

	//Calculate
	for(int weight_group = 0; weight_group < amount_weight_group; weight_group++)
		sum =+ this->layer[layer].weight[weight_group][index] * input[weight_group];

	return 1/(1+exp(-(sum + this->layer[layer].theta[index])));
}


int main()
{
	srand(time(NULL));
	//Var
	int cos_param;
	float cos_target;
	int sin_param;
	float sin_target;

	float ** input  = (float **) calloc(DATA_SIZE, sizeof(float *));
	float ** target = (float **) calloc(DATA_SIZE, sizeof(float *));
	//Network setup
	int layer_size[DEPTH] = {3, 3, PARAM_SIZE};
	NNClass * NN = new NNClass(DEPTH, CONSTANT, PARAM_SIZE, layer_size);

	//Set data
	for(int data = 0; data < DATA_SIZE; data++)
	{
		cos_param = rand()%360;
		sin_param = rand()%360;
		cos_target = cos(cos_param*PI/180);
		sin_target = sin(sin_param*PI/180);

		input[data]  = (float *) calloc(PARAM_SIZE, sizeof(float));
		target[data] = (float *) calloc(PARAM_SIZE, sizeof(float));

		input [data][0] = (float)cos_param;
		input [data][1] = (float)sin_param;
		target[data][0] = cos_target;
		target[data][1] = sin_target;
	}

	//Start a training;
	NN->train(input, target, DATA_SIZE);
	NN->destroy();

	//Free data
	for(int data = 0; data < DATA_SIZE; data++)\
	{
		free(input[data]);
		free(target[data]);
	}
	free(input);
	free(target);
}
