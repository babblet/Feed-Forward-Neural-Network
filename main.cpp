//** Notes ** 
//  fix misspellings
//  Make it backprop work
//  Implement error checking
//  Organisize

#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

bool write_to_serial(int iteration ,std::vector<float> input, std::vector<float> target, std::vector<float> output)
{
	std::cout << "iter = " << iteration << " | ";

	for(int index = 0; index < input.size(); index++)
		std::cout << "Input[" << index << "] = "<< input[index] << std::endl;

	std::cout << "target = ";
	for(int index = 0; index < target.size(); index++)
		std::cout << target[index];

	std::cout << " | output = ";
	for(int index = 0; index < output.size(); index++)
		std::cout << output[index];
	
	std::cout << std::endl;
	return 0;
}

//Used in randomize_weigths();
float random_weight(){
	//Return random float between -1 and 1
	srand(time(NULL));
	return ((float)rand() / ((float)RAND_MAX / 2)) - 1; 
}

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

NNClass::NNClass(int depth, float constant, int input_size, std::vector<int> layer_size)
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

void NNClass::train(std::vector<std::vector<float> > input, std::vector<std::vector<float> > target, int iterations)
{
	for(int data = 0; data < iterations; data++)
	{
		for(int layer = 0; layer < this->depth; layer++)
		{
			
			//float * layer_input;
			//if(layer > 0) layer_input = this->layer[layer - 1].output;
			//else          layer_input = input[data];

			for(int index = 0; index < this->layer_size[layer]; index++)
			{
				if(layer > 0)
				{ 
					this->layer[layer].output[index] = this->activation(layer, index, this->layer[layer - 1].output);
					this->backpropagation(this->layer[layer - 1].output, target[data]);
				}
				else
				{
					this->layer[layer].output[index] = this->activation(layer, index, input[data]);
					this->backpropagation(input[data], target[data]);
				}
			}
		}
		write_to_serial(data ,input[data], target[data], this->layer[this->depth - 1].output);
	}
}

bool NNClass::allocate_layers()
{
	//Allocate layer pointer array of set depth
	this->layer.resize(this->depth);
	
	//Allocate layers.
	for(int layer = 0; layer < this->depth; layer++)
	{
		int amount_weight_groups;
		if(layer > 0) amount_weight_groups = this->layer_size[layer - 1]; // Rest of layers
		else          amount_weight_groups = this->input_size;            // Input layer

		//Weight groups
		this->layer[layer].weight.resize(amount_weight_groups, std::vector<float>(this->layer_size[layer])); 

		//Weights | set to 0
		//for(int weight_group = 0; weight_group < amount_weight_groups; weight_group++)
		//{
		//	this->layer[layer].weight[weight_group].resize(this->layer_size[layer]); 
		//	std::cout << weight_group << " weight_group" << std::endl;
		//}
			
		//Neurons | set to 0
		this->layer[layer].theta.resize(this->layer_size[layer], 0); 
		this->layer[layer].delta.resize(this->layer_size[layer], 0);  
		this->layer[layer].output.resize(this->layer_size[layer], 0);
	}

	return true;
}

//Is it needed to free Vectors??
/*bool NNClass::destroy()
{

	for(int layer = 0; layer < this->depth; layer++)
	{
		int amount_weight_group;
		if(layer > 0) amount_weight_group = this->layer_size[layer - 1];
		else          amount_weight_group = this->input_size;

		for(int weight_group = 0; weight_group < amount_weight_group; weight_group++)
			free(this->layer[layer].weight[weight_group]);
		
		std::cout << layer << "| 1" << std::endl;
		free(this->layer[layer].theta);
		std::cout << layer << "| 2" << std::endl;
		free(this->layer[layer].delta);
		std::cout << layer << "| 3" << std::endl;
		free(this->layer[layer].output);
	}
	std::cout << "Free done" << std::endl;
	return true;
}
*/


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
				this->layer[layer].weight[weight_group][weight] = random_weight();
			}
		}
	}
	return true;
}

//Com
bool NNClass::backpropagation(std::vector<float> input, std::vector<float> target)
{
	//Setup pointers for easier reading??
	
	//Update
	for(int index = 0; index < layer_size[this->depth - 1]; index++)
		this->layer[this->depth - 1].delta[index] = 
			(target[index] - this->layer[this->depth - 1].output[index]) * 
			this->layer[this->depth - 1].output[index] * 
			(1 - this->layer[this->depth - 1].output[index]);

	//Hidden delta update
	for(int layer = this->depth - 2; layer >= 0; layer--)
	{
		//Update
		for(int index = 0; index < this->layer_size[layer]; index++)
		{
			float sum = 0;
			//Calculate sum of the previous deltas and weights
			for(int weight_index = 0; weight_index < this->layer_size[layer + 1]; weight_index++)
				sum += this->layer[layer + 1].delta[weight_index] * this->layer[layer + 1].weight[index][weight_index];

			//Calculate delta for current layer and group
			this->layer[layer].delta[index] = this->layer[layer].output[index] * (1 - this->layer[layer].output[index]) * sum;
		}
	}

	//Weight and Theta update
	for(int layer = 0; layer < this->depth; layer++)
	{
		int amount_weight_group;
		std::vector<float> layer_input;
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
				this->layer[layer].weight[weight_group][index] += 
					(this->constant * this->layer[layer].delta[index] * layer_input[weight_group]);

			this->layer[layer].theta[index] += (this->constant * this->layer[layer].delta[index]);
		}
	}
	return true;
}
float NNClass::activation(int layer, int index, std::vector<float> input)
{
	
	//Setup
	float sum = 0;
	int amount_weight_group;

	if(layer > 0) amount_weight_group = this->layer_size[layer - 1];
	else          amount_weight_group = this->input_size;

	//Calculate
	for(int weight_group = 0; weight_group < amount_weight_group; weight_group++)
		sum =+ this->layer[layer].weight[weight_group][index] * input[weight_group];

	//Sigmoid activation
	//return 1/(1+exp(-(sum + this->layer[layer].theta[index])));

	//Step activation | Threshold activation
	if(sum > 0) return 1.0f;
	else 	    return 0.0f;
}


#define INPUT_SIZE 1
#define DEPTH 4
#define OUTPUT_SIZE 8
#define DATA_SIZE 1000000

#include <bitset>
std::string binary(int decimal)
{
    return std::bitset<8>(decimal).to_string(); //to binary
}

//Deciam to binary network
int main()
{
	srand(time(NULL));
	//Var
	int decimal;

	std::string string_target;

	std::vector<std::vector<float>> input(DATA_SIZE); 
	std::vector<std::vector<float>> target(DATA_SIZE); 
	
	//Network setup
	std::vector<int> layer_size(DEPTH);
	layer_size = {3, 3, 3, 8};

	NNClass NN(DEPTH, 1000, INPUT_SIZE, layer_size);

	//Create Data
	for(int data = 0; data < DATA_SIZE; data++)
	{
		decimal = (rand() % 256);
		string_target = binary(decimal);

		input[data].resize(INPUT_SIZE); 
		target[data].resize(OUTPUT_SIZE); 

		input [data][0] = decimal;
		
		//Hardcode testing
		target[data][0] = string_target[0] - '0'; 
		target[data][1] = string_target[1] - '0'; 
		target[data][2] = string_target[2] - '0'; 
		target[data][3] = string_target[3] - '0'; 
		target[data][4] = string_target[4] - '0'; 
		target[data][5] = string_target[5] - '0'; 
		target[data][6] = string_target[6] - '0'; 
		target[data][7] = string_target[7] - '0'; 
	}
	
	//Start a training;
	NN.train(input, target, DATA_SIZE);
	//NN.destroy();

	//Free data vetors???

	return 0;
}
