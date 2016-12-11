#include "waveletFilter.h"

//#define SIGNAL_LENGTH_2D 16384 
#define SIGNAL_LENGTH_2D 32
//#define SIGNAL_LENGTH_2D 16
#define COMPRESSION_LEVELS_2D 1

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

//low inverse filters
double * host_low_inverse_filter_array_2D = 0;
double * device_low_inverse_filter_array_2D = 0;

//high inverse filters
double * host_high_inverse_filter_array_2D = 0;
double * device_high_inverse_filter_array_2D = 0;

//reconstructed signal
double * host_reconstructed_signal_array_2D = 0;
double * device_reconstructed_signal_array_2D = 0;

//tmp memory
double * deviceTmpMemory = 0;

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

void initReconstructedSignal2D() {

    int64 signalLength = get1DSignalLength();
    int64 num_bytes = signalLength * sizeof(double);
    assert(num_bytes != 0);

    host_reconstructed_signal_array_2D = (double*)malloc(num_bytes);

    cudaError_t err = cudaMalloc((void**)&device_reconstructed_signal_array_2D, num_bytes);
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
    int64 num_bytes = lowFilterLenght * sizeof(double);

    host_low_filter_array_2D = (double*)malloc(num_bytes);

    filter2D.getLowPassFilter(host_low_filter_array_2D);

    cudaMalloc((void**)&device_low_filter_array_2D, num_bytes);

    cudaMemcpy(device_low_filter_array_2D, host_low_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initHighFilter_2D() {
    int64 highFilterLenght = 9;
    int64 num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array_2D = (double*)malloc(num_bytes);

    filter2D.getHighPassFilter(host_high_filter_array_2D);
    cudaMalloc((void**)&device_high_filter_array_2D, num_bytes);

    cudaMemcpy(device_high_filter_array_2D, host_high_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initLowInverseFilter_2D() {
    int64 lowFilterInverseLenght = 9;
    int64 num_bytes = lowFilterInverseLenght * sizeof(double);

    host_low_inverse_filter_array_2D = (double*)malloc(num_bytes);

    filter2D.getLowReconstructFilter(host_low_inverse_filter_array_2D);

    cudaMalloc((void**)&device_low_inverse_filter_array_2D, num_bytes);

    cudaMemcpy(device_low_inverse_filter_array_2D, host_low_inverse_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initHighInverseFilter_2D() {
    int64 highInverseFilterLenght = 9;
    int64 num_bytes = highInverseFilterLenght * sizeof(double);

    host_high_inverse_filter_array_2D = (double*)malloc(num_bytes);

    filter2D.getHighReconstructFilter(host_high_inverse_filter_array_2D);
    cudaMalloc((void**)&device_high_inverse_filter_array_2D, num_bytes);

    cudaMemcpy(device_high_inverse_filter_array_2D, host_high_inverse_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initOutput_2D() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&device_output_array_2D, num_bytes);
    if(err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
}

void initDeviceTmpMemory() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    assert(num_bytes != 0);
    cudaMalloc((void**)&deviceTmpMemory, num_bytes);
}
    

void transferMemoryBack_2D() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    assert(num_bytes != 0);

    host_output_array_2D = (double*)malloc(num_bytes);
    cudaMemcpy(host_output_array_2D, device_output_array_2D, num_bytes, cudaMemcpyDeviceToHost);
}

void transferReconstructedSignalBack_2d() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    assert(num_bytes != 0);
    cudaMemcpy(host_reconstructed_signal_array_2D, device_reconstructed_signal_array_2D, num_bytes, cudaMemcpyDeviceToHost);
}

void printResult_2D(double * signal) {
    int64 signalLenght = get1DSignalLength();
    int64 stride = SIGNAL_LENGTH_2D;
    
    for(int64 i = 0;i < signalLenght;i++) {
        if(i % stride == 0) {
            std::cerr<<std::endl;
        }
        std::cerr<<signal[i]<<" ";
    }
    std::cerr<<std::endl;
}

void test2D() {
    std::cerr<<"Testing 2D Decompose"<<std::endl;
    filter2D.constructFilters();
    initLowFilter_2D();
    initHighFilter_2D();
    initLowInverseFilter_2D();
    initHighInverseFilter_2D();
    initOutput_2D();

    initSignal2D();
    copyInputSignal2D();
    initReconstructedSignal2D();
    initDeviceTmpMemory();
    //decompose the signal 
    MyVector levels;

    struct ImageMeta imageMeta;
    imageMeta.imageWidth = SIGNAL_LENGTH_2D;
    imageMeta.imageHeight = SIGNAL_LENGTH_2D;
    imageMeta.xStart = 0;
    imageMeta.yStart = 0;
    imageMeta.xEnd = SIGNAL_LENGTH_2D;
    imageMeta.yEnd = SIGNAL_LENGTH_2D;

    auto startDecompose = std::chrono::system_clock::now();
    dwt2D(levels, COMPRESSION_LEVELS_2D, device_signal_array_2D,
                    imageMeta, device_low_filter_array_2D,
                    device_high_filter_array_2D, 9, imageMeta,
                    device_output_array_2D, deviceTmpMemory);
    auto endDecompose = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = endDecompose-startDecompose;
    std::cout<< diff.count() << " s\n";
    transferMemoryBack_2D();

    //do 
    //{
        //std::cout << '\n' << "Press a key to continue...";
    //} while (std::cin.get() != '\n');

    printResult_2D(host_output_array_2D);

    int levelUncompress = 1;
    iDwt2D(levels, levelUncompress,
                      device_output_array_2D, 
                      imageMeta,
                      device_low_inverse_filter_array_2D,
                      device_high_inverse_filter_array_2D,
                      9,
                      device_reconstructed_signal_array_2D,
                      deviceTmpMemory);
    return;
    transferReconstructedSignalBack_2d();
    printResult_2D(host_reconstructed_signal_array_2D);
}
