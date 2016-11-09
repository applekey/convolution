#include "waveletFilter.h"
#include <stdio.h>
#define SIGNAL_LENGTH 256

//signal
double * host_signal_array = 0;
double * device_signal_array = 0;

//output
double * device_output_array = 0;

//low filters
double * host_low_filter_array = 0;
double * device_low_filter_array = 0;

//high filters
double * host_high_filter_array = 0;
double * device_high_filter_array = 0;

waveletFilter filter;

__global__ void convolveWavelet(double * filter, int filterLength, 
                                double * inputSignal, int signalLength,
                                double * output, int outputOffset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int inputIndex = index * 2 + (filterLength - 1); 

    double sum = 0.0;

    for(int i = 0; i < filterLength; i++) {
        sum += filter[i] * inputSignal[inputIndex];
    }

    output[index + outputOffset] = sum; 
}


/*void init(double * signal, int signalLength, double * filter, int filterLength) {*/
double * initSignal() {
    int signalLenght = SIGNAL_LENGTH;
    int num_bytes = signalLenght * sizeof(double);

    host_signal_array = (double*)malloc(num_bytes);
    cudaMalloc((void**)&device_signal_array, num_bytes);

    cudaMemcpy(device_signal_array, host_signal_array, num_bytes, cudaMemcpyHostToDevice);
    return device_signal_array;
}

double * initOutput() {
    int outputLenght = SIGNAL_LENGTH;
    int num_bytes = outputLenght * sizeof(double);

    cudaMalloc((void**)&device_output_array, num_bytes);
    return device_output_array;
}

double * initLowFilter() {
    int lowFilterLenght = 9;
    int num_bytes = lowFilterLenght * sizeof(double);

    host_low_filter_array = (double*)malloc(num_bytes);

    filter.getLowPassFilter(host_low_filter_array);

    cudaMalloc((void**)&device_low_filter_array, num_bytes);

    cudaMemcpy(device_low_filter_array, host_low_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_low_filter_array;
}

double * initHighFilter() {
    int highFilterLenght = 9;
    int num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array = (double*)malloc(num_bytes);

    filter.getLowPassFilter(host_high_filter_array);
    cudaMalloc((void**)&device_high_filter_array, num_bytes);

    cudaMemcpy(device_high_filter_array, host_high_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_high_filter_array;
}

void init() {
    initLowFilter();
    initHighFilter();
    initSignal();
    initOutput();
}

void freeMemory() {
    free(host_signal_array);

    cudaFree(device_signal_array);
    cudaFree(device_output_array);

    free(host_low_filter_array);
    cudaFree(device_low_filter_array);

    free(host_high_filter_array);
    cudaFree(device_high_filter_array);
}

int main(int argc, const char * argv[]) {
    //generate constant signal that is power of 2, 64
    int signalLength = 64;
    double inputSignal[64];

    for(int i = 0; i < 64; i++) {
        inputSignal[0] = 1.0;
    }

    filter.constructFilters();
    init();

    int block_size = 128;
    int gridSize = SIGNAL_LENGTH / block_size;
    //convolve high filters
    int outputOffset = 0;
    convolveWavelet<<<gridSize, block_size>>>(device_high_filter_array, 9, 
                    device_signal_array,SIGNAL_LENGTH,
                    device_output_array, 0);

    outputOffset = SIGNAL_LENGTH / 2;
    //convolve low filters
    convolveWavelet<<<gridSize, block_size>>>(device_low_filter_array, 9, 
                    device_signal_array,SIGNAL_LENGTH,
                    device_output_array, outputOffset);
    //transfer output back
    //done free memory 
    freeMemory();

    return 0;
}
