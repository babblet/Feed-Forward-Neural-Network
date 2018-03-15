#include "NNClass.h"
#include <bitset>

#define INPUT_SIZE 1
#define DEPTH 4
#define OUTPUT_SIZE 8
#define DATA_SIZE 1000000

//int to binary string
std::string binary(int decimal) { return std::bitset<8>(decimal).to_string(); }

//Decimal to binary network
int main()
{
	srand(time(NULL));

	int decimal;
	std::string string_target;
	std::vector<std::vector<float>> input(DATA_SIZE); 
	std::vector<std::vector<float>> target(DATA_SIZE); 
	
	//Network setup
	std::vector<int> layer_size(DEPTH);
	layer_size = {3, 3, 3, 8};

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
	
	//Start training;
	NNClass NN(input, target, layer_size, 0.005);
	NN.train();
	
	//NN.destroy();

	//Free data vetors???

	return 0;
}
