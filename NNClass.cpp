//** Notes **
//  Make more compact
//  Is saving the output needed?? Cost function looks like it will be needed for proper calculations. But it is an sum so probably it is not needed.
//  Fix misspellings
//  Make the backpropagation work
//  Implement error checking

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

	//Set bias to 1;
	for(int data = 0; data < this->data_size; data++)
		for(int layer = 0; layer < this->depth - 2; layer++)
			this->layer[layer].output[data][this->layer_size[layer] - 1] = 1;

}

//NNClass::load(string filepath){};
//NNClass::save(string path, string filename){};

void NNClass::train()
{
	for(int data = 0; data < this->data_size; data++)
	{
		for(int layer = 1; layer < this->depth; layer++)
			for(int index = 0; index < this->layer_size[layer]; index++)
			{
				//float act = activation(layer, index, data);
				//std::cout << "layer = " << layer << " | index = " << index << " | activation = " << act << std::endl;
				this->layer[layer].output[data][index] = activation(layer, index, data);
			}
		backpropagation(data);
		
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


bool NNClass::backpropagation(int data)
{
	//Output Delta
	for(int index = 0; index < this->layer_size[this->depth - 1]; index++)
		this->layer[this->depth - 1].delta[index] = this->layer[this->depth - 1].output[data][index] - this->target[data][index];


	//Hidden delta
	for(int layer = this->depth - 2; layer > 0; layer--)
	{
		std::vector<float> sum(this->layer_size[layer], 0);
		for(int weight = 0; weight < this->layer_size[layer + 1]; weight++)
		{
			for(int index = 0; index < this->layer_size[layer]; index++)
			{
				sum[index] += (this->layer[layer].weight[index][weight] * this->layer[layer + 1].delta[weight]);				
				
			}

		}

		for(int index = 0; index < this->layer_size[layer]; index++)
		{
			this->layer[layer].delta[index] = sum[index] * this->layer[layer].output[data][index];
		}

		std::vector<std::vector<float> > weight_error;
		weight_error.resize(this->layer_size[layer], std::vector<float>(this->layer_size[layer + 1]));		
		//comp
		for(int index = 0; index < this->layer_size[layer]; index++)
		{	
			if(index > 0)
			{
				for(int weight = 0; weight < this->layer_size[layer + 1]; weight++)
				{
					weight_error[index][weight] = (1.0f/2.0f) * (this->layer[layer].output[data][index] * this->layer[layer + 1].delta[weight] + this->layer[layer].weight[index][weight]);
				}
			}
			else
			{ 
				for(int weight = 0; weight < this->layer_size[layer + 1]; weight++)
				{
					weight_error[index][weight] = (1.0f/2.0f) * (this->layer[layer].output[data][index] * this->layer[layer + 1].delta[weight]);
				}
			}
		}
		
		//update weights;
		for(int index = 0; index < this->layer_size[layer]; index++)
		{
			for(int weight = 0; weight < this->layer_size[layer + 1]; weight++)
			{
				float grad = -(this->constant * weight_error[index][weight]);
				//std::cout << "grad = " << grad << std::endl;
				this->layer[layer].weight[index][weight] += grad;
				
			}
		}
	}

	return true;
}

//bool NNClass::cost()
//{
//	return true;
//}

float NNClass::activation(int layer, int weight, int data)
{	
	//Setup
	float sum = 0;
	//Calculate
	for(int index = 0; index < this->layer_size[layer - 1]; index++)
	{
		float x,y;
 		y = this->layer[layer - 1].weight[index][weight];
		x = this->layer[layer - 1].output[data][index];
	
//		std::cout << "x = " << x << std::endl;
//		std::cout << "y = " << y << std::endl;
		sum = sum + (x * y);
	}

//	std::cout << "sum = " << sum << std::endl;
	//Sigmoid activation
	return 1/(1+exp(-(sum)));

	//Step activation | Threshold activation
	//if(sum > 0) return 1.0f;
	//else 	    return 0.0f;
}
