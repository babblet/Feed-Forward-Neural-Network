//** Notes **
//  Fix destructor
//  Make more compact
//  Implement error checking

#include "NNClass.h"

//This is temp
bool write(float cost, int iteration, int epoch, std::vector<float> input, std::vector<float> target, std::vector<float> output)
{
	std::cout << "epoch = " << epoch << " \t| iter = " << iteration << " \t| ";

	std::cout << "Input = "<< input[0] << std::endl;

	std::cout << "target = ";
	for(int index = 0; index < target.size(); index++)
	{
		std::cout << target[index];
	}
	std::cout << " \t| output = ";
	
	for(int index = 0; index < output.size(); index++)
	{
		std::cout << output[index] << "\t, ";
	}
	std::cout << "\t\t cost = " << cost << std::endl;

	return 0;
}

NNClass::NNClass(int data_size, std::vector<int> &layer_size)
{
	//data	
	this->data_size = data_size;

	//layers
	for(int index = 0; index < layer_size.size(); index++)
	{
		layer_size[index] += 1;
	}
	this->layer_size = layer_size;
	this->depth = this->layer_size.size();
	
	//Alloc
	this->allocate_layers();

	//Setup
	this->randomize_weights();
	for(int data = 0; data < this->data_size; data++)
	{
		for(int layer = 0; layer < this->depth - 1; layer++)
		{
			this->layer[layer].output[data][this->layer_size[layer] - 1] = 1;
		}
	}
}

NNClass::NNClass(int data_size, std::string &filename) // Load 
{
	std::ifstream file(filename);
	std::string str;
	std::vector<std::string> data;
	int p, weight, index, layer;

	while(std::getline(file, str))
	{
		data.push_back(str);
	}

	//Get structure
	p = 0;
	for(int index = 0; index < data[0].length(); index++)
	{
		if(data[0][index] == '\n') break;
		else if(data[0][index] == ' ')
		{
			this->layer_size.push_back(atoi(data[0].substr(p, index - p).c_str()));
			p = index + 1;
		}
	}

	this->depth = this->layer_size.size();
	this->data_size = data_size;
	this->allocate_layers();

	//Get weights
	p = 0;
	weight = 0;
	index = 0;
	layer = 0;
	for(int data_index = 0; data_index < data[1].length(); data_index++)
	{
		if(data[1][data_index] == '\n') break;
		else if(data[1][data_index] == ' ')
		{
			this->layer[layer].weight[index][weight] = atof(data[1].substr(p, data_index - p).c_str());
			p = data_index + 1;

			weight++;
			if(weight == this->layer_size[layer + 1])
			{
				weight = 0;
				index++;
				if(index == this->layer_size[layer])
				{
					index = 0;
					layer++;
				}
			}
		}
	}
	file.close();
}

void NNClass::save(std::string filename)
{
	std::ofstream file;
	file.open (filename);
	
	
	//save structure
	for(int layer = 0; layer < this->depth; layer++)
	{
		file << this->layer_size[layer] << " ";
	}
	file << "\n";

	//save weights
	for(int layer = 0; layer < this->depth - 1; layer++)
	{
		for(int index = 0; index < this->layer_size[layer]; index++)
		{
			for(int weight = 0; weight < this->layer_size[layer + 1]; weight++)
			{				
				file << this->layer[layer].weight[index][weight] << " ";
			}
		}
	}
	file.close();
}

void NNClass::train(std::vector<std::vector<float> > &input, std::vector<std::vector<float> > &target, float constant, int epochs)
{
	srand(time(NULL));
	this->constant = constant;
	this->target = target;
	this->layer[0].output = input;
	float cost_v = 0;

	for(int epoch = 0; epoch < epochs; epoch++)
	{
		int random = (rand() % 1000);
		for(int data = 0; data < this->data_size; data++)
		{
			//Feedforward
			for(int layer = 1; layer < this->depth; layer++)
			{
				for(int index = 0; index < this->layer_size[layer] - 1; index++)
				{	
					this->layer[layer].output[data][index] = activation(layer, index, data);
				}
			}
			backpropagation(data);

			if(data%(10000 + random) == 0)
			{ 	
				write(cost_v, data, epoch, this->layer[0].output[data], this->target[data], this->layer[this->depth - 1].output[data]);
			}
		}
		cost_v = cost();
	}
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
			this->layer[layer].weight.resize(this->layer_size[layer], std::vector<float>(this->layer_size[layer + 1] - 1)); 
		
		this->layer[layer].delta.resize(this->layer_size[layer] - 1, 0);
		this->layer[layer].output.resize(this->data_size, std::vector<float>(this->layer_size[layer]));
	}
	return true;
}

bool NNClass::randomize_weights()
{
	srand(time(NULL));

	//Set random weights
	for(int layer = 0; layer < this->depth - 1; layer++)
	{
		for(int weight_group = 0; weight_group < this->layer_size[layer]; weight_group++)
		{
			for(int weight = 0; weight < this->layer_size[layer + 1] - 1; weight++)
			{
				this->layer[layer].weight[weight_group][weight] = ((float)rand() / (RAND_MAX / 1.0f)) - 1.0f; //Random 2 to -2
			}
		}
	}
	return true;
}

bool NNClass::backpropagation(int data)
{
	//Output layer/delta
	for(int index = 0; index < this->layer_size[this->depth - 1] - 1; index++)
	{
		this->layer[this->depth - 1].delta[index] = this->layer[this->depth - 1].output[data][index] - this->target[data][index];
	}
	
	//Hidden layers
	for(int layer = this->depth - 2; layer > 0; layer--)
	{
		//Hidden delta
		if(layer > 0)
		{
			std::vector<float> sum(this->layer_size[layer], 0);
			for(int weight = 0; weight < this->layer_size[layer + 1] - 1; weight++) // -1 remove bias
			{
				for(int index = 0; index < this->layer_size[layer] - 1; index++) // -1 remove bias
				{
					sum[index] += (this->layer[layer].weight[index][weight] * this->layer[layer + 1].delta[weight]);				
				}
			}
			for(int index = 0; index < this->layer_size[layer]; index++)
			{
				this->layer[layer].delta[index] = sum[index] * this->layer[layer].output[data][index] * (1.0f - this->layer[layer].output[data][index]);
			}
		}
		
		//Update weights
		for(int weight_group = 0; weight_group < this->layer_size[layer]; weight_group++)
		{
			for(int weight = 0; weight < this->layer_size[layer + 1] - 1; weight++) // -1 remove bias
			{	this->layer[layer].weight[weight_group][weight] += ( 
					-this->constant * 
					this->layer[layer + 1].delta[weight] * 
					this->layer[layer].output[data][weight_group] *
					(1.0f - this->layer[layer].output[data][weight_group]) * 
					this->layer[layer].output[data][weight_group]
				);
			}		
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
		{	
			sum = sum + (1.0f/this->data_size) * pow(this->target[data][index] - this->layer[this->depth - 1].output[data][index], 2);
		}
	}
	return sum;
}

float NNClass::activation(int layer, int weight, int data)
{	
	float sum = 0;
	for(int index = 0; index < this->layer_size[layer - 1]; index++)
	{
		sum = sum + (this->layer[layer - 1].output[data][index] * this->layer[layer - 1].weight[index][weight]);
	}
	//Sigmoid activation
	return 1.0f/(1.0f+exp(-(sum)));
	
	
	//TanH activation
	//return tanh(sum); 

	//Step activation | Threshold activation
	//if(sum > 0) return 1.0f;
	//else 	    return 0.0f;
}
