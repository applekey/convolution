#include <stdio.h>
#define SIGNAL_LENGTH 256

__device__ void convolveWavelet(double * filter, int filterLength, 
                                double * inputSignal, int inputIndex,
                                double * output, int outputIndex) {
}

__global__ void kernel(double *array)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  array[index] ++;
}

/*void init(double * signal, int signalLength, double * filter, int filterLength) {*/
double * initSignal() {
    int signalLenght = SIGNAL_LENGTH;
    int num_bytes = signalLenght * sizeof(double);

    double * host_signal_array = 0;
    double * device_signal_array = 0;

    host_signal_array = (double*)malloc(num_bytes);
    cudaMalloc((void**)&device_signal_array, num_bytes);

    cudaMemcpy(device_signal_array, host_signal_array, num_bytes, cudaMemcpyHostToDevice);
    return device_signal_array;
}

double * initOutput() {
    int outputLenght = SIGNAL_LENGTH;
    int num_bytes = outputLenght * sizeof(double);

    double * device_output_array = 0;
    cudaMalloc((void**)&device_output_array, num_bytes);
    return device_output_array;
}

double * initLowFilter() {
    int lowFilterLenght = 9;
    int num_bytes = lowFilterLenght * sizeof(double);

    double * host_low_filter_array = 0;
    double * device_low_filter_array = 0;

    host_low_filter_array = (double*)malloc(num_bytes);
    cudaMalloc((void**)&device_low_filter_array, num_bytes);

    cudaMemcpy(device_low_filter_array, host_low_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_low_filter_array;
}

double * initHighFilter() {
    int highFilterLenght = 9;
    int num_bytes = highFilterLenght * sizeof(double);

    double * host_high_filter_array = 0;
    double * device_high_filter_array = 0;

    host_high_filter_array = (double*)malloc(num_bytes);
    cudaMalloc((void**)&device_high_filter_array, num_bytes);

    cudaMemcpy(device_high_filter_array, host_high_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_high_filter_array;
}

void init() {
    initLowFilter();
    initHighFilter();
    initSignal();
}



/*
void init() {
    int signalLength = SIGNAL_LENGTH;

    int num_bytes = signalLength * sizeof(double);
    // copy the signal into the device
    double *device_array = 0;
    double *host_array = 0;

    // malloc a host signal array
    host_array = (double*)malloc(num_bytes);
    //intilize host array with signal

    for(int i = 0; i < signalLength; i++) {
        host_array[i] = 7.0;
    }

    // cudaMalloc a device array
    cudaMalloc((void**)&device_array, num_bytes);

    int block_size = 128;
    int grid_size = signalLength / block_size;

    //copy input signal to device
    cudaMemcpy(device_array, host_array, num_bytes, cudaMemcpyHostToDevice);

    kernel<<<grid_size,block_size>>>(device_array);

    //download and inspect the result on the host:
    cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

    // print out the result element by element
    for(int i=0; i < signalLength; ++i) {
        printf("%f ", host_array[i]);
    }

    // deallocate memory
    free(host_array);
    cudaFree(device_array);
    
}

*/

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
