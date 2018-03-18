#include "NNClass.h"
#include <cmath>

#define INPUT_SIZE 1
#define DEPTH 5
#define OUTPUT_SIZE 1
#define DATA_SIZE 100000
#define PI 3.14159
#define EPOCH 100

//sin function test
int main()
{
	srand(time(NULL));

	int angle;
	std::string string_target;
	std::vector<std::vector<float>> input(DATA_SIZE); 
	std::vector<std::vector<float>> target(DATA_SIZE); 
	
	//Network setup
	std::vector<int> hidden_layer_size(DEPTH);
	hidden_layer_size = {3, 10, 3,OUTPUT_SIZE}; //temp output

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
	NNClass NN(input, target, hidden_layer_size, 0.05f , EPOCH);
	NN.train();
	return 0;
}
