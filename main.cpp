#include "NNClass.hpp"
#include <cmath>

#define INPUT_SIZE 1
#define DEPTH 5
#define OUTPUT_SIZE 1
#define DATA_SIZE 100000
#define PI 3.14159
#define EPOCH 500
#define CONSTANT 0.005f
#define FILENAME "sin_weights"

//  sin function test
int main()
{
	srand(time(NULL));

	int angle;
	std::string filename = FILENAME;
	std::string string_target;
	std::vector<std::vector<float> > input(DATA_SIZE); 
	std::vector<std::vector<float> > target(DATA_SIZE); 
	
	//Network setup
	std::vector<int> layer_size(DEPTH);
       	int layer_size_content[] = {INPUT_SIZE, 3, 10, 3, OUTPUT_SIZE};
	layer_size.insert(layer_size.begin(), layer_size_content, layer_size_content+DEPTH);

	//Create Data
	for(int data = 0; data < DATA_SIZE; data++)
	{
		angle = rand() % 179 + 1;

		input[data].resize(INPUT_SIZE); 
		target[data].resize(OUTPUT_SIZE); 

		//Hardcode testinge
		input[data][0] = ((float)angle*(float)PI/180);		
		target[data][0] = sin((float)angle*(float)PI/180); 
	}
	
	//Start training;
	//NNClass NN(DATA_SIZE, layer_size);
	//NN.train(input, target, CONSTANT, EPOCH);
	//NN.save(filename);
	//NN.destroy();

	//Reload training
	NNClass NN2(DATA_SIZE, filename);
	NN2.train(input, target, CONSTANT, EPOCH);
	NN2.save(filename);
	//NN2.destroy();
	
	return 0;
}
