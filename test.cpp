#include "NNClass.h"
#include <cmath>
#include <time.h>

#define DEPTH 3
#define PARAM_SIZE 2
#define CONSTANT 0.05
#define PI 3.14159265
#define DATA_SIZE 50


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
	
//	cout << NN->output(input) << endl;

	NN->destroy();



	//Free data
	for(int data = 0; data < DATA_SIZE; data++)\
	{
		free(input[data]);
		free(target[data]);
	}
	free(input);
	free(target);
	return 0;	
}
