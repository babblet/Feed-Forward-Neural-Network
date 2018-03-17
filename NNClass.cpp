//** Notes **
//  Is saving the output needed?? Cost function looks like it will be needed for proper calculations. But it is an sum so probably it is not needed.
//  fix misspellings
//  Make the backpropagation work
//  Implement error checking
//  Organisize

#include "NNClass.h"

//This is temp
bool write(int iteration ,std::vector<float> input, std::vector<float> target, std::vector<float> output)
{
	std::cout << "iter = " << iteration << " | ";

	for(int index = 0; index < input.size(); index++)
		std::cout << "Input[" << index << "] = "<< input[index] << std::endl;

	std::cout << "target = ";
	for(int index = 0; index < target.size(); index++)
		std::cout << target[index];

	std::cout << " | output = ";
	for(int index = 0; index < output.size(); index++)
		std::cout << output[index] << ", ";
	std::cout << std::endl;

	return 0;
}

NNClass::NNClass(std::vector<std::vector<float> > &input, std::vector<std::vector<float> > &target, std::vector<int> &hidden_layer_size, float constant)
{
	//Add bias to hidden layers
	for(int index = 0; index < hidden_layer_size.size() - 1; index++)
		hidden_layer_size[index] += 1;

	this->data_size = input.size();
	this->input_size = input[0].size() + 1;
	
	this->target = target;

	this->layer_size.push_back(this->input_size);
	this->layer_size.insert(std::end(this->layer_size), std::begin(hidden_layer_size), std::end(hidden_layer_size));
	this->depth = this->layer_size.size();

	this->constant = constant;
	
	this->allocate_layers();
	this->layer[0].output = input;

	this->randomize_weigths();
}

//NNClass::load(string filepath){};
//NNClass::save(string path, string filename){};

void NNClass::train()
{
	std::cout << this->depth << std::endl;
	for(int data = 0; data < this->data_size; data++)
	{
		for(int layer = 1; layer < this->depth; layer++)
			for(int index = 0; index < this->layer_size[layer]; index++)
				this->layer[layer].output[data][index] = activation(layer, index, this->layer[layer - 1].output[data]);

		backpropagation();

		if(data%1000 == 0) 
			write(data ,this->layer[0].output[data], this->target[data], this->layer[this->depth - 1].output[data]);
	}
//	std::cout << cost() << std::endl;
}

bool NNClass::allocate_layers()
{
	//Allocate layer pointer array of set depth
	this->layer.resize(this->depth);
	
	//Allocate layers.
	for(int layer = 0; layer < this->depth; layer++)
	{
		//Weight groups
		if(layer < this->depth - 1)
		{
			this->layer[layer].weight.resize(this->layer_size[layer], std::vector<float>(this->layer_size[layer + 1])); 
			this->layer[layer].bias.resize(this->layer_size[layer], 1); 
		}
		
		this->layer[layer].delta.resize(this->layer_size[layer], 0);  
		this->layer[layer].output.resize(this->data_size, std::vector<float>(this->layer_size[layer]));
	}
	return true;
}

bool NNClass::randomize_weigths()
{
	srand(time(NULL));

	//Set random weigths
	for(int layer = 0; layer < this->depth - 1; layer++)
		for(int weight_group = 0; weight_group < this->layer_size[layer]; weight_group++)
			for(int weight = 0; weight < this->layer_size[layer + 1]; weight++)
				this->layer[layer].weight[weight_group][weight] = ((float)rand() / ((float)RAND_MAX / 2)) - 1; //Random 1 to -1

	return true;
}


bool NNClass::backpropagation()
{
	
	return true;
}

//bool NNClass::cost()
//{
//	return true;
//}

float NNClass::activation(int layer, int index, std::vector<float> input)
{	
	//Setup
	float sum = 0;
	//Calculate
	for(int weight_group = 0; weight_group < this->layer_size[layer - 1]; weight_group++)
		sum =+ this->layer[layer - 1].weight[weight_group][index] * input[weight_group];

	//Sigmoid activation
	return 1/(1+exp(-(sum)));

	//Step activation | Threshold activation
	//if(sum > 0) return 1.0f;
	//else 	    return 0.0f;
}
