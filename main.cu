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


__device__ void convolveWavelet(double * filter, int filterLength, 
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
    cudaMalloc((void**)&device_low_filter_array, num_bytes);

    cudaMemcpy(device_low_filter_array, host_low_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_low_filter_array;
}

double * initHighFilter() {
    int highFilterLenght = 9;
    int num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array = (double*)malloc(num_bytes);
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

int main(int argc, const char * argv[]) {
    //generate constant signal that is power of 2, 64
    int signalLength = 64;
    double inputSignal[64];

    for(int i = 0; i < 64; i++) {
        inputSignal[0] = 1.0;
    }

    init();
    /*init(inputSignal, inputSignal, );*/

    return 0;
}
