#define SIGNAL_PAD_VALUE 1.0
#include <iostream>
#include <vector>
#include <cassert>

#define MAX_X 1024

int calculateCoefficientLength(std::vector<int> &L, int levels,
                                int inputSignalLength) {

    int totalLength = 0;
    int currentCoefficientLength = inputSignalLength / 2; //assume that all signals are powers of 2
    L.resize(levels + 2); //+ 2 levels, 1 is the final low coefficients and the last as an end bookeeping
    L[0] = 0;

    for(int i = 1 ; i < levels; i++) {
        L[i] = currentCoefficientLength + L[i-1];
        currentCoefficientLength /= 2;
    }

    //append last level, which is just the high and low coefficients
    L[levels] = currentCoefficientLength + L[levels-1];
    L[levels+1] = currentCoefficientLength + L[levels];
    totalLength = L[levels + 1];
    return totalLength;
}

__global__ void convolveWavelet(double * filter, int filterLength, 
                                double * inputSignal, int signalLength,
                                double * output, int outputOffset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int inputIndex = index * 2 + (filterLength / 2); 

    double sum = 0.0;

    for(int i = 0; i < filterLength; i++) {
        sum += filter[i] * inputSignal[inputIndex - (filterLength / 2) + i];
    }

    output[index + outputOffset] = sum; 
}

__global__ void extend(double * inputSignal, int signalLength, int filterLength,
                       double * extendedSignal) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int sideWidth = filterLength / 2;

    if(index >= signalLength + sideWidth * 2) {
        return;
    }

    if(index < sideWidth) {

        extendedSignal[index] = inputSignal[0];

    } else if(index < sideWidth + signalLength) {

        extendedSignal[index] = inputSignal[index - sideWidth];

    }  else {

        //extendedSignal[index] = SIGNAL_PAD_VALUE;
        extendedSignal[index] = inputSignal[0];
    } 
}

__global__ void inverseConvolve(double * lowReconstructFilter, double * highReconstructFilter,
                                int filterLength, 
                                double * highLowCoefficients, double * reconstructedSignal,
                                int signalLength) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= signalLength) {
        return;
    }

    double sum = 0.0;
    int lowIndex, highIndex;

    if( index % 2 != 0) {
        lowIndex = filterLength - 2; 
        highIndex = filterLength - 1;
    } else {
        lowIndex = filterLength - 1; 
        highIndex = filterLength - 2;
    }
    //sum low
    int lowCoefficientIndex = (index + 1) / 2;
    while(lowCoefficientIndex > -1) {
        sum += lowReconstructFilter[lowCoefficientIndex]; 
        lowCoefficientIndex -= 2;
    }

    //sum high
    int highCoefficientIndex = (index) / 2;
    while(highCoefficientIndex > -1) {
        sum += highReconstructFilter[highCoefficientIndex]; 
        highCoefficientIndex -= 2;
    }
    //write out sum
    reconstructedSignal[index] = sum;
}

double * initTmpCoefficientMemory(int signalLength) {
    double * lowCoefficientMemory = 0; 
    long long num_bytes = signalLength * sizeof(double);
    assert(num_bytes != 0);
    cudaMalloc((void**)&lowCoefficientMemory, num_bytes);
    return lowCoefficientMemory;
}

void calculateBlockSize(int totalLength, 
                        int & x, int & y) {
    
    if(totalLength > MAX_X) {
        x = MAX_X;
        int extra = totalLength % MAX_X;
        if(extra != 0) {
            y = totalLength /  MAX_X + 1;
        } else {
            y = totalLength /  MAX_X;
        }
    } else {
        x = totalLength;
        y = 1;
    }
    std::cerr<<"given "<<totalLength<<" dims are:"<<x<<":"<<y<<std::endl;
}

void debugLowMemory(double * deviceMem, int length) {
    std::cerr<<"Debugging Low memory"<<std::endl;
    length = 10;
    long long num_bytes = length * sizeof(double);

    double * tmp = (double*)malloc(num_bytes);
    cudaMemcpy(tmp, deviceMem, num_bytes, cudaMemcpyDeviceToHost);  

    for(int i = 0; i< length; i++) {
        std::cerr<<tmp[i]<<std::endl;
    } 
    delete [] tmp;
    std::cerr<<"Debugging Low memory Stop"<<std::endl;
}

void Idwt(std::vector<int> & L, int levelsToReconstruct, 
          int signalLength, 
          double * coefficients) {
    int maxExtendedSignalLength = signalLength;
    //allocate
    int currentCoefficientIndex = L.size(); 
    double * extendedHighCoeff;
    double * extendedLowCoeff;
    
    for(int i = 1; i < levelsToReconstruct; i++) {
        
        //extend signals
    }
}

void dwt(std::vector<int> & L, int levelsToCompress,
         double * deviceInputSignal, int signalLength,
         double * deviceLowFilter, 
         double * deviceHighFilter,
         double * deviceOutputCoefficients,
         int filterLength) {

    int gridSize = 1;
    int currentSignalLength = signalLength;
    int currentHighCoefficientOffset = 0 + signalLength / 2; 
    //create a tempory low coefficient / signal extend array
     
    int inputSignalExtendedLength = currentSignalLength + (filterLength / 2 ) * 2;

    double * deviceLowCoefficientMemory = initTmpCoefficientMemory(inputSignalExtendedLength);
    double * currentDeviceSignal = deviceInputSignal;

    for(int level = 0; level < levelsToCompress; level++) {

        //convolve high filters
        int outputOffset = 0;

        //extend the signal
        int inputSignalExtendedLength = currentSignalLength + (filterLength / 2 ) * 2;

        int xThread = -1;
        int yThread = -1;
        calculateBlockSize(inputSignalExtendedLength, xThread, yThread);

        extend<<<yThread, xThread>>>(currentDeviceSignal, currentSignalLength, 
                            filterLength, deviceLowCoefficientMemory);

        //debugLowMemory(deviceLowCoefficientMemory, inputSignalExtendedLength);
        ////convolve low filters
        int block_size = currentSignalLength / 2;

        int lowCoeffOffset = 0;
        if(level == levelsToCompress - 1) {
            lowCoeffOffset = L[level + 1] + signalLength / 2;
        }

        calculateBlockSize(block_size, xThread, yThread);
        convolveWavelet<<<yThread, xThread>>>(deviceLowFilter, filterLength, 
                        deviceLowCoefficientMemory, inputSignalExtendedLength,
                        deviceOutputCoefficients, lowCoeffOffset);
        
        ////convolve high filters
        convolveWavelet<<<yThread, xThread>>>(deviceHighFilter, filterLength, 
                        deviceLowCoefficientMemory, inputSignalExtendedLength,
                        deviceOutputCoefficients, currentHighCoefficientOffset);


        currentSignalLength /= 2;
        currentHighCoefficientOffset = L[level + 1] + signalLength / 2;
        currentDeviceSignal = deviceOutputCoefficients;
    }
    //finally copy the low coefficients to the end 
        
    //free tmp memory
    cudaFree(deviceLowCoefficientMemory);
}
