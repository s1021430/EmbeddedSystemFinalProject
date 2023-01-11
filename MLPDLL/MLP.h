#pragma once

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>

using namespace std;

#ifndef		_MLP_H
#define		_MLP_H
#endif

#ifndef BYTE
#define BYTE unsigned char
#endif

#define NUMBER_OF_LAYERS 3
#define NUMBER_OF_INPUT_NODES 150
#define NUMBER_OF_OUTPUT_NODES 16
#define MAXIMUM_LAYERS 250
#define MAXIMUM_NUMBER_OF_SETS 100
#define EPOCHS 600
#define ERROR_THRESHOLD 0.0002
#define LEARNING_RATE 150
#define SLOPE 0.014
#define WEIGHT_BIAS 30

class MLP {
private:
	void initialize_weights();
	void form_input_set(BYTE** samples, int num);
	void form_desired_output_set(char* trainer_string);
	void train_network();
	void form_network();
	void get_inputs(int set_number);
	void get_desired_outputs(int set_number);
	void calculate_outputs();
	double sigmoid(double f_net);
	double sigmoid_derivative(double result);
	int threshold(double val);
	void calculate_errors();
	float get_average_error();
	void calculate_weights();
	void character_to_unicode(char* character);
	void write_text_to_log_file(const std::string &text);
public:
	bool load_network(char* network_load_file_name);
	string classify(BYTE* sample);
	bool save_network(char* network_save_file);
	bool training(BYTE** samples, char* trainer_string, int num);
};
