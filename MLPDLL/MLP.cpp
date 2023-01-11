#include "MLP.h"

#include <codecvt>
#include <fstream>
#include <ostream>
#include <vector>
#include <bitset>
#include <iostream>
#include <locale>
#include <algorithm>

using namespace std;

int layers[NUMBER_OF_LAYERS];
int number_of_input_sets = 0;
bool training_flag = false;
double current_input[NUMBER_OF_INPUT_NODES] = {};
double input_set[NUMBER_OF_INPUT_NODES][MAXIMUM_NUMBER_OF_SETS] = {};
int desired_output[NUMBER_OF_OUTPUT_NODES] = {};
int desired_output_set[NUMBER_OF_OUTPUT_NODES][MAXIMUM_NUMBER_OF_SETS] = {};
double node_output[NUMBER_OF_LAYERS][MAXIMUM_LAYERS] = {};
double weight[NUMBER_OF_LAYERS][MAXIMUM_LAYERS][MAXIMUM_LAYERS] = {};
double error[NUMBER_OF_LAYERS][MAXIMUM_LAYERS] = {};
int output_bit[NUMBER_OF_OUTPUT_NODES] = {};
int desired_output_bit[NUMBER_OF_OUTPUT_NODES] = {};

bool MLP::load_network(char* network_load_file_name)
{
    // Open a text file for writing
    write_text_to_log_file("load_network");
    //ifstream network_load_file_stream(network_load_file_name);
    ifstream inFile;
    inFile.open(network_load_file_name);
    form_network();
    string weight_text;
    string line;
    vector<char> weight_char(20);
    int title_length, weight_length;

    for (int i = 0; i < 9; i++) {
        std::getline(inFile, line);
    }

    for (int i = 1; i < NUMBER_OF_LAYERS; i++) {
        for (int j = 0; j < layers[i]; j++) {
            for (int k = 0; k < layers[i - 1]; k++) {
                weight_text = "";
                std::getline(inFile, line);
                title_length = ("Weight[" + std::to_string(i) + " , " + std::to_string(j) + " , " + std::to_string(k) + "] = ").length();
                weight_length = line.length() - title_length;
                std::copy(line.begin() + title_length, line.end(), weight_char.begin());

                for (int counter = 0; counter < weight_length; counter++) {
                    weight_text += weight_char[counter];
                }
                
                weight[i][j][k] = std::stod(weight_text);
            }
        }
    }
    training_flag = true;
    inFile.close();
    return true;
}

void MLP::form_network()
{
    layers[0] = NUMBER_OF_INPUT_NODES;
    layers[NUMBER_OF_LAYERS - 1] = NUMBER_OF_OUTPUT_NODES;
    for (int i = 1; i < NUMBER_OF_LAYERS - 1; i++)
    {
        layers[i] = MAXIMUM_LAYERS;
    }
}

void MLP::initialize_weights()
{
    srand(time(nullptr));
    for (int i = 1; i < NUMBER_OF_LAYERS; i++)
        for (int j = 0; j < layers[i]; j++)
            for (int k = 0; k < layers[i - 1]; k++)
                weight[i][j][k] = (double)(rand() / RAND_MAX) * (WEIGHT_BIAS - -WEIGHT_BIAS) + -WEIGHT_BIAS;
}

void MLP::form_input_set(BYTE** samples, int num)
{
    number_of_input_sets = num;
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < number_of_input_sets; j++)
        {
            /*if (samples[j, i] > 0)
                  input_set[j, i] = 1;
              else
                  input_set[j, i] = 0;*/

            input_set[j][i] = samples[j][i] / 255.0;
        }
    }
}

void MLP::form_desired_output_set(char* trainer_string)
{
    for (int i = 0; i < number_of_input_sets; i++) {
        character_to_unicode(trainer_string);
        for (int j = 0; j < NUMBER_OF_OUTPUT_NODES; j++) {
            desired_output_set[j][i] = desired_output_bit[j];
        }
    }
}

void MLP::train_network()
{
    srand(time(nullptr));
	for (int epoch = 0; epoch <= EPOCHS; epoch++) {
        float average_error = 0.0;
        for (int i = 0; i < number_of_input_sets; i++) {
	        const int set_number = rand() % number_of_input_sets;
            //set_number = i;
            get_inputs(set_number);
            get_desired_outputs(set_number);
            calculate_outputs();
            calculate_errors();
            calculate_weights();
            average_error = average_error + get_average_error();
        }

        average_error = average_error / number_of_input_sets;
        if (average_error < ERROR_THRESHOLD) {
            epoch = EPOCHS + 1;
        }
    }
}

string MLP::classify(BYTE* sample)
{
    if (training_flag == false) {
        return "";
    }
    const size_t number_of_input_sets = 1;

    for (size_t i = 0; i < NUMBER_OF_INPUT_NODES; i++) {
        input_set[i][0] = sample[i] / 255.0;
    }
    get_inputs(0);
    for (int i = 0; i < NUMBER_OF_INPUT_NODES; i++)
        write_text_to_log_file("current_input[i] = " + to_string(current_input[i]));
    calculate_outputs();

    for (int i = 0; i < NUMBER_OF_LAYERS; i++) {
        for (int j = 0; j < layers[i]; j++) {
            write_text_to_log_file(to_string(node_output[i][j]));
        }
    }

    int dec = 0;
    for (size_t i = 0; i < 8; i++) {
        output_bit[i] = threshold(node_output[NUMBER_OF_LAYERS - 1][i]);
        dec += output_bit[i] * (int)(std::pow(2, i));
    }
    const char output_char = static_cast<char>(dec);
    write_text_to_log_file("dec = " + to_string(output_char));
    return std::string(1, output_char);
}

bool MLP::save_network(char* network_save_file)
{
    ofstream network_save_file_stream(network_save_file);
    network_save_file_stream << "MLP Weight values. 2022 NTUT Embedded Machine Vision. " << std::endl;
    network_save_file_stream << "Network Name	= OCR Test" << std::endl;
    network_save_file_stream << "Hidden Layer Size	= " << MAXIMUM_LAYERS << std::endl;
    network_save_file_stream << "Number of Patterns= " << number_of_input_sets << std::endl;
    network_save_file_stream << "Number of Epochs	= " << EPOCHS << std::endl;
    network_save_file_stream << "Learning Rate	= " << LEARNING_RATE << std::endl;
    network_save_file_stream << "Sigmoid Slope	= " << SLOPE << std::endl;
    network_save_file_stream << "Weight Bias	= " << WEIGHT_BIAS << std::endl;
    network_save_file_stream << "" << std::endl;

    for (int i = 1; i < NUMBER_OF_LAYERS; i++) {
        for (int j = 0; j < layers[i]; j++) {
            for (int k = 0; k < layers[i - 1]; k++) {
                network_save_file_stream << "Weight[" << i << " , " << j << " , " << k << "] = ";
                network_save_file_stream << weight[i][j][k] << std::endl;
            }
        }
    }
    network_save_file_stream.close();
    return true;
}

bool MLP::training(BYTE** samples, char* trainer_string, int num)
{
    form_network();
    initialize_weights();
    form_input_set(samples, num);
    form_desired_output_set(trainer_string);

    train_network();

    training_flag = true;

    return true;
}

void MLP::get_inputs(int set_number)
{
    for (int i = 0; i < NUMBER_OF_INPUT_NODES; i++)
        current_input[i] = input_set[i][set_number];
}

void MLP::get_desired_outputs(int set_number)
{
    for (int i = 0; i < NUMBER_OF_INPUT_NODES; i++)
        desired_output[i] = desired_output_set[i][set_number];
}

void MLP::calculate_outputs() {
	int number_of_weights = 0;
    for (int i = 0; i < NUMBER_OF_LAYERS; i++) {
        for (int j = 0; j < layers[i]; j++) {
            double f_net = 0.0;
            if (i == 0) number_of_weights = 1;
            else number_of_weights = layers[i - 1];
            
            for (int k = 0; k < number_of_weights; k++) {
                if (i == 0)
                    f_net = current_input[j];
                else
                    f_net = f_net + node_output[i - 1][k] * weight[i][j][k];
            }

            node_output[i][j] = sigmoid(f_net);
        }
    }
}

double MLP::sigmoid(double f_net) {
    //double result = 1.0 / (1.0 + exp(-1 * slope * f_net));    //Unipolar
    const double result = (2.0 / (1.0 + exp(-1 * SLOPE * f_net))) - 1; //Bipolar
    return result;
}

double MLP::sigmoid_derivative(double result)
{
    //float derivative=(float)(result*(1-result));					//Unipolar
    const double derivative = 0.5F * (1 - pow(result, 2));           //Bipolar			
    return derivative;
}

int MLP::threshold(double val)
{
    if (val < 0.5)
        return 0;
    return 1;
}

void MLP::calculate_errors() {
	for (int i = 0; i < NUMBER_OF_OUTPUT_NODES; i++)
        error[NUMBER_OF_LAYERS - 1][i] = (sigmoid_derivative(node_output[NUMBER_OF_LAYERS - 1][i]) * (desired_output[i] - node_output[NUMBER_OF_LAYERS - 1][i]));

    for (int i = NUMBER_OF_LAYERS - 2; i >= 0; i--) {
        for (int j = 0; j < layers[i]; j++) {
            double sum = 0.0;
            for (int k = 0; k < layers[i + 1]; k++)
                sum = sum + error[i + 1][k] * weight[i + 1][k][j];
            error[i][j] = (sigmoid_derivative(node_output[i][j]) * sum);
        }
    }
}

float MLP::get_average_error() {
    double average_error = 0.0;
    for (int i = 0; i < NUMBER_OF_OUTPUT_NODES; i++)
        average_error = average_error + error[NUMBER_OF_LAYERS - 1][i];
    average_error = average_error / NUMBER_OF_OUTPUT_NODES;
    return std::abs(average_error);
}

void MLP::calculate_weights() {
    for (int i = 1; i < NUMBER_OF_LAYERS; i++)
        for (int j = 0; j < layers[i]; j++)
            for (int k = 0; k < layers[i - 1]; k++) {
                weight[i][j][k] = weight[i][j][k] + LEARNING_RATE * error[i][j] * node_output[i - 1][k];
            }
}

void MLP::character_to_unicode(char* character) {
    wstring_convert<codecvt_utf8<wchar_t>> converter;
    wstring_convert<codecvt_utf8<wchar_t>> conv;
    wstring wstr = conv.from_bytes(character);
    
    string bytes = converter.to_bytes(wstr);

    size_t byte_count = bytes.size();
    copy_n(wstr.data(),byte_count,bytes.begin());
    bitset<8> bits(bytes);
    const int bit_array_length = bytes.length();
    for (int i = 0; i < bit_array_length; i++)
    {
        if (bits[i])
            desired_output_bit[i] = 1;
        else
            desired_output_bit[i] = 0;
    }
}

void MLP::write_text_to_log_file( const std::string &text )
{
    std::ofstream log_file(
        "log.txt", std::ios_base::out | std::ios_base::app );
    log_file << text << std::endl;
}