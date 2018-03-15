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

NNClass::NNClass(std::vector<std::vector<float> > &input, std::vector<std::vector<float> > &target, std::vector<int> &layer_size, float constant)
{
	this->input = input;
	this->data_size = input.size();
	this->input_size = input[0].size();
	
	this->target = target;

	this->layer_size = layer_size;
	this->depth = layer_size.size();

	this->constant = constant;
	
	this->allocate_layers();
	this->randomize_weigths();
}

//NNClass::load(string filepath){};
//NNClass::save(string path, string filename){};

void NNClass::train()
{
	for(int data = 0; data < this->data_size; data++)
	{
		for(int layer = 0; layer < this->depth; layer++)
		{
			
			std::vector<float> layer_input;
			if(layer > 0) layer_input = this->layer[layer - 1].output[data];
			else          layer_input = this->input[data];

			for(int index = 0; index < this->layer_size[layer]; index++)
					this->layer[layer].output[data][index] = this->activation(layer, index, layer_input);
						
		}
		this->backpropagation(this->layer[this->depth - 1].output[data], this->target[data], data);
		
		if(data%1000 == 0)
			write(data ,input[data], target[data], this->layer[this->depth - 1].output[data]);
	}
	std::cout << cost() << std::endl;
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
		
		//Neurons | set to 0
		this->layer[layer].theta.resize(this->layer_size[layer], 0); 
		this->layer[layer].delta.resize(this->layer_size[layer], 0);  
		this->layer[layer].output.resize(this->data_size, std::vector<float>(this->layer_size[layer]));
	}
	return true;
}

bool NNClass::randomize_weigths()
{
	srand(time(NULL));
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
				this->layer[layer].weight[weight_group][weight] = ((float)rand() / ((float)RAND_MAX / 2)) - 1; //Random 1 to -1
		}
	}
	return true;
}

//Com										  //fix this
bool NNClass::backpropagation(std::vector<float> output, std::vector<float> target, int data)
{
	//Setup pointers for easier reading??
	
	//Update
	for(int index = 0; index < layer_size[this->depth - 1]; index++)
		this->layer[this->depth - 1].delta[index] = 
			(target[index] - this->layer[this->depth - 1].output[data][index]) * 
			this->layer[this->depth - 1].output[data][index] * 
			(1 - this->layer[this->depth - 1].output[data][index]);

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
			this->layer[layer].delta[index] = this->layer[layer].output[data][index] * (1.0f - this->layer[layer].output[data][index]) * sum;
		}
	}

	//Weight and Theta update
	for(int layer = 0; layer < this->depth; layer++)
	{
		int amount_weight_group;
		std::vector<float> layer_output;
		if(layer > 0){
			amount_weight_group = this->layer_size[layer - 1];
			layer_output = this->layer[layer - 1].output[data]; // <-- yes this...
		}
		else{
			amount_weight_group = this->input_size;
			layer_output = output;
		}

		//Update
		for(int index = 0; index < this->layer_size[layer]; index++)
		{
			for(int weight_group = 0; weight_group < amount_weight_group; weight_group++)
				this->layer[layer].weight[weight_group][index] += 
					(this->constant * this->layer[layer].delta[index] * layer_output[weight_group]);

			this->layer[layer].theta[index] += (this->constant * this->layer[layer].delta[index]);
		}
	}
	return true;
}

float NNClass::cost()
{
	float sum = 0;
	for(int data = 0; data < this->data_size; data++)
	{ 
		for(int index = 0; index < this->layer_size[this->depth - 1]; index++)
			sum += pow(this->target[data][index] - this->layer[this->depth - 1].output[data][index], 2);
	}	
	return (1/(2*(this->data_size)))*sum;
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
	return 1/(1+exp(-(sum + this->layer[layer].theta[index])));

	//Step activation | Threshold activation
	//if(sum > 0) return 1.0f;
	//else 	    return 0.0f;
}
