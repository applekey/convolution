#include "waveletFilter.h"

#define SIGNAL_LENGTH_2D 4 
#define COMPRESSION_LEVELS_2D 10

#include "helper2D.h"

//signal
double * host_signal_array_2D = 0;
double * device_signal_array_2D = 0;

//low filters
double * host_low_filter_array_2D = 0;
double * device_low_filter_array_2D = 0;

//high filters
double * host_high_filter_array_2D = 0;
double * device_high_filter_array_2D = 0;

//output
double * host_output_array_2D = 0;
double * device_output_array_2D = 0;

int64 get1DSignalLength() {
    int64 totalSignalLength = SIGNAL_LENGTH_2D * SIGNAL_LENGTH_2D;
    assert(totalSignalLength > SIGNAL_LENGTH_2D); //check for overflow
    return totalSignalLength;
}

waveletFilter filter2D;

void initSignal2D() {

    int64 signalLength = get1DSignalLength();
    int64 num_bytes = signalLength * sizeof(double);
    assert(num_bytes != 0);

    host_signal_array_2D = (double*)malloc(num_bytes);

    for(int64 i = 0; i < signalLength; i++) {
        /*host_signal_array[i] = 1.0 * sin((double)i /100.0) * 100.0;*/
        host_signal_array_2D[i] = 1.0;
    }
}

void copyInputSignal2D() {

    int64 num_bytes = get1DSignalLength() * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&device_signal_array_2D, num_bytes);

    if(err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
    cudaMemcpy(device_signal_array_2D, host_signal_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initLowFilter_2D() {
    int64 lowFilterLenght = 9;
    int64 num_bytes = get1DSignalLength() * sizeof(double);

    host_low_filter_array_2D = (double*)malloc(num_bytes);

    filter2D.getLowPassFilter(host_low_filter_array_2D);

    cudaMalloc((void**)&device_low_filter_array_2D, num_bytes);

    cudaMemcpy(device_low_filter_array_2D, host_low_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initHighFilter_2D() {
    int64 highFilterLenght = 9;
    int64 num_bytes = get1DSignalLength() * sizeof(double);

    host_high_filter_array_2D = (double*)malloc(num_bytes);

    filter2D.getHighPassFilter(host_high_filter_array_2D);
    cudaMalloc((void**)&device_high_filter_array_2D, num_bytes);

    cudaMemcpy(device_high_filter_array_2D, host_high_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initOutput_2D() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&device_output_array_2D, num_bytes);
    if(err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
}

void test2D() {
    std::cerr<<"Testing 2D Decompose"<<std::endl;
    filter2D.constructFilters();
    initLowFilter_2D();
    initHighFilter_2D();
    initOutput_2D();

    initSignal2D();
    copyInputSignal2D();
    //decompose the signal 
    MyVector levels;

    struct ImageMeta imageMeta;
    imageMeta.imageWidth = SIGNAL_LENGTH_2D;
    imageMeta.imageHeight = SIGNAL_LENGTH_2D;
    imageMeta.xStart = 0;
    imageMeta.yStart = 0;
    imageMeta.xEnd = SIGNAL_LENGTH_2D;
    imageMeta.yEnd = SIGNAL_LENGTH_2D;

    dwt2D_Horizontal(levels, COMPRESSION_LEVELS_2D, device_signal_array_2D,
                    imageMeta, device_low_filter_array_2D,
                    device_high_filter_array_2D, 9, imageMeta,
                    device_output_array_2D);
}
