#include "waveletFilter.h"
#include <stdio.h>
#define SIGNAL_LENGTH 256

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
    int extendedInputSignalLength = signalLenght + (9 - 1) * 2;
    int num_bytes = extendedInputSignalLength * sizeof(double);

    host_signal_array = (double*)malloc(num_bytes);

    for(int i = 0; i < extendedInputSignalLength; i++) {
        host_signal_array[i] = 1.0;
    }

    for(int i = 0; i < 9 - 1; i++) {
        host_signal_array[i] = 0.0;
    }
    
    for(int i = 0; i < 9 - 1; i++) {
        host_signal_array[extendedInputSignalLength - 1 - i] = 0.0;
    }

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

void transferMemoryBack() {
    int num_bytes = SIGNAL_LENGTH * sizeof(double);

    host_output_array = (double*)malloc(num_bytes);
    cudaMemcpy(host_output_array, device_output_array, num_bytes, cudaMemcpyDeviceToHost);  
    //print output
    for(int i = 0; i < SIGNAL_LENGTH ;i ++) {
        printf("%f \n", host_output_array[i]);
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

int main(int argc, const char * argv[]) {
    filter.constructFilters();
    init();

    int block_size = 128;
    int gridSize = SIGNAL_LENGTH / block_size;
    //convolve high filters
    int outputOffset = 0;
    int inputSignalExtendedLength = SIGNAL_LENGTH + (9 - 1) * 2;
    convolveWavelet<<<gridSize, block_size>>>(device_high_filter_array, 9, 
                    device_signal_array, inputSignalExtendedLength,
                    device_output_array, 0);

    outputOffset = SIGNAL_LENGTH / 2;
    //convolve low filters
    convolveWavelet<<<gridSize, block_size>>>(device_low_filter_array, 9, 
                    device_signal_array, inputSignalExtendedLength,
                    device_output_array, outputOffset);
    //transfer output back
    transferMemoryBack();
    //done free memory 
    freeMemory();

    return 0;
}
