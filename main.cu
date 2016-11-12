#include "waveletFilter.h"
#include "helper.h"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>

/*#define SIGNAL_LENGTH 536870912 */
#define SIGNAL_LENGTH  32 
#define COMPRESSION_LEVELS 4

using namespace std;

//signal
double * host_signal_array = 0;
double * device_signal_array = 0;

//output
double * host_output_array = 0;
double * device_output_array = 0;

//low filters
double * host_low_filter_array = 0;
double * device_low_filter_array = 0;

//high filters
double * host_high_filter_array = 0;
double * device_high_filter_array = 0;

waveletFilter filter;

std::vector<int> coefficientIndicies; 

double * initSignal() {

    long num_bytes = SIGNAL_LENGTH * sizeof(double);

    host_signal_array = (double*)malloc(num_bytes);

    for(int i = 0; i < SIGNAL_LENGTH; i++) {
        /*host_signal_array[i] = 1.0 * sin((double)i /100.0) * 100.0;*/
        host_signal_array[i] = 1.0;
    }

    cudaError_t err = cudaMalloc((void**)&device_signal_array, num_bytes);

    if(err != cudaSuccess){
         printf("The error is %s", cudaGetErrorString(err));
    }
    cudaMemcpy(device_signal_array, host_signal_array, num_bytes, cudaMemcpyHostToDevice);
    return device_signal_array;
}

double * initOutput(int outputLength) {
    long num_bytes = outputLength * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&device_output_array, num_bytes);
    if(err != cudaSuccess){
         printf("The error is %s", cudaGetErrorString(err));
    }
    return device_output_array;
}

double * initLowFilter() {
    int lowFilterLenght = 9;
    long num_bytes = lowFilterLenght * sizeof(double);

    host_low_filter_array = (double*)malloc(num_bytes);

    filter.getLowPassFilter(host_low_filter_array);

    cudaMalloc((void**)&device_low_filter_array, num_bytes);

    cudaMemcpy(device_low_filter_array, host_low_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_low_filter_array;
}

double * initHighFilter() {
    int highFilterLenght = 9;
    long num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array = (double*)malloc(num_bytes);

    filter.getHighPassFilter(host_high_filter_array);
    cudaMalloc((void**)&device_high_filter_array, num_bytes);

    cudaMemcpy(device_high_filter_array, host_high_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_high_filter_array;
}

void transferMemoryBack(int outputLength) {
    long num_bytes = outputLength * sizeof(double);

    host_output_array = (double*)malloc(num_bytes);
    cudaMemcpy(host_output_array, device_output_array, num_bytes, cudaMemcpyDeviceToHost);  
}

void printOutputCoefficients(double * hostOutput, std::vector<int> coefficientIndicies) {
    int offset = SIGNAL_LENGTH / 2;
    int coefficientLevels = coefficientIndicies.size();

    /*int total = coefficientIndicies[3];*/
    /*std::cerr<<coefficientLevels<<" "<<total<<std::endl;*/
    /*for(int i =0; i< total; i++) {*/
        /*std::cerr<<hostOutput[offset + i]<<std::endl;*/
    /*}*/
    
    for(int i = 0; i < coefficientLevels - 1;i++) {
        std::cerr<<"Level: "<<i<<std::endl;
        int levelCoefficientIndex = coefficientIndicies[i];
        int numberOfCoefficents = coefficientIndicies[i + 1] - coefficientIndicies[i];

        for(int j = 0; j < numberOfCoefficents; j++) {
            double coeffVal = hostOutput[levelCoefficientIndex + j + offset];
            std::cerr<<coeffVal<<" ";
        }
        std::cerr<<std::endl;
    }
}

void freeMemory() {
    free(host_signal_array);
    free(host_output_array);

    cudaFree(device_signal_array);
    cudaFree(device_output_array);

    free(host_low_filter_array);
    cudaFree(device_low_filter_array);

    free(host_high_filter_array);
    cudaFree(device_high_filter_array);
}

void writeResultsToMemory(double * output, int length) {
    double epsilon = 0.0000001;
    double a = -1.41442e-12;
    double b = 1.41421;

    int offset = SIGNAL_LENGTH / 2;
    for(int i = 0; i < length/2; i++) {
        if(abs(a -  output[i + offset]) < epsilon * 1.0e-12 ) {
            std::cerr<<"error "<<output[i + offset]<<std::endl;
        }
    }
    for(int i = length/2; i < length; i++) {
        if(abs(b -  output[i + offset]) < epsilon) {
            std::cerr<<"error "<<output[i + offset]<<std::endl;
        }
    }
    return;
    /*int offset = SIGNAL_LENGTH / 2;*/
    /*ofstream myfile;*/
    /*myfile.open("output.txt");*/
    
    /*for(int i = 0; i < length; i++) {*/
        /*myfile << output[i + offset]<<"\n";*/
    /*}*/
    /*myfile.close();*/
}


int main(int argc, const char * argv[]) {
    int outputLength = calculateCoefficientLength(coefficientIndicies, COMPRESSION_LEVELS, SIGNAL_LENGTH);
    outputLength += SIGNAL_LENGTH / 2; //add extra for buffer for first low coefficient

    filter.constructFilters();
    initLowFilter();
    initHighFilter();
    initSignal();
    initOutput(outputLength);

auto start = std::chrono::system_clock::now();
    //run filter   
    dwt(coefficientIndicies, COMPRESSION_LEVELS, 
        device_signal_array, SIGNAL_LENGTH,
        device_low_filter_array, device_high_filter_array,
        device_output_array, 9);

    //transfer output back
    transferMemoryBack(outputLength);

auto end = std::chrono::system_clock::now();
std::chrono::duration<double> diff = end-start;
std::cout<< diff.count() << " s\n";
    printOutputCoefficients(host_output_array, coefficientIndicies);

    /*int ab = calculateCoefficientLength(coefficientIndicies, COMPRESSION_LEVELS, SIGNAL_LENGTH);*/
    /*writeResultsToMemory(host_output_array, ab);*/

    //done free memory 
    freeMemory();

    return 0;
}
