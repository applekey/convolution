#ifndef HELPER_H
#define HELPER_H

#define SIGNAL_PAD_VALUE 1.0
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>   

#define MAX_X 1024
#define MAX_Y 4096

typedef long long int64;
struct MyVector {
    int64 currentSize = 0;
    int64 indicies[1000];

    int64 & operator[](int idx) {
        return indicies[idx];
    }

    void resize(int64 newSize) {
        currentSize = newSize;
    }

    int size() {
        return currentSize;
    }
};

int calculateCoefficientLength(MyVector &L, int levels,
                               int64 inputSignalLength) {

    int64 totalLength = 0;
    int64 currentCoefficientLength = inputSignalLength / 2; //assume that all signals are powers of 2
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

__device__ int64 calculateIndex() {
    int64 blockId = blockIdx.y * gridDim.x + blockIdx.x;
    return blockId * blockDim.x + threadIdx.x;
}

__global__ void convolveWavelet(double * filter, int64 filterLength,
                                double * inputSignal, int64 signalLength,
                                double * output, int64 outputOffset) {
    int64 index = calculateIndex();
    if(index >= signalLength) {
        return;
    }
    int64 inputIndex = index * 2 + (filterLength / 2);
    // load into shared memory
    //__shared__ double s[1024 + 8]; //max per
    //s[threadIdx.x] = inputSignal[inputIndex];

    //if(threadIdx.x == 0) {
        //s[1024] = 1.0; 
        //s[1025] = 1.0; 
        //s[1026] = 1.0; 
        //s[1027] = 1.0; 
        //s[1028] = 1.0; 
        //s[1029] = 1.0; 
        //s[1030] = 1.0; 
        //s[1031] = 1.0; 
        //s[1032] = 1.0; 
    //}

    //__syncthreads();    
     

    double sum = 0.0;

    for(int64 i = 0; i < filterLength; i++) {
        //sum += filter[i] * s[threadIdx.x + i];

        sum += filter[i] * inputSignal[inputIndex - (filterLength / 2) + i];
    }

    output[index + outputOffset] = sum;
}

__global__ void extend(double * inputSignal, int64 signalLength, int64 filterLength,
                       double * extendedSignal) {

    int64 index = calculateIndex();

    int64 sideWidth = filterLength / 2;

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
                                int64 filterLength,
                                double * lowCoefficients, double * highCoefficients,
                                double * reconstructedSignal,
                                int64 signalLength) {
    int64 index = calculateIndex();
    if(index >= signalLength) {
        return;
    }

    double sum = 0.0;
    int64 lowIndex, highIndex;

    lowIndex = filterLength - 1;
    highIndex = filterLength - 2;
    //sum low
    int64 lowCoefficientIndex = (index + 1) / 2;
    while(lowIndex > -1) {
        sum += lowReconstructFilter[lowIndex] * lowCoefficients[lowCoefficientIndex];
        lowIndex -= 2;
        lowCoefficientIndex++;
    }

    //sum high
    int64 highCoefficientIndex = (index) / 2;
    while(highIndex > -1) {
        sum += highReconstructFilter[highIndex] * highCoefficients[highCoefficientIndex];
        highIndex -= 2;
        highCoefficientIndex++;
    }

    //write out sum
    reconstructedSignal[index] = sum;
}
void calculateBlockSize(int64 totalLength,
                        int & threads, dim3 & blocks) {

    if(totalLength > MAX_X) {
        threads = MAX_X;
        if(totalLength / MAX_X > MAX_Y) {
            int64 extra = totalLength % (MAX_X *MAX_Y);
            if(extra != 0) {
                blocks.x = MAX_Y;
                blocks.y = totalLength / (MAX_X *MAX_Y) + 1;
                blocks.z = 1;
            } else {
                blocks.x = MAX_Y;
                blocks.y = totalLength / (MAX_X *MAX_Y);
                blocks.z = 1;
            }
            //
        } else {
            int64 extra = totalLength % MAX_X;
            if(extra != 0) {
                int y = totalLength /  MAX_X + 1;
                blocks.x = y;
                blocks.y = 1;
                blocks.z = 1;
            } else {
                int y = totalLength /  MAX_X;
                blocks.x = y;
                blocks.y = 1;
                blocks.z = 1;
            }
        }
    } else {
        threads = totalLength;
        blocks.x = 1;
        blocks.y = 1;
        blocks.z = 1;
    }
    //std::cerr<<"block: "<<blocks.x<<" "<<blocks.y<<" "<<blocks.z<<std::endl;
}

void debugTmpMemory(double * deviceMem, int64 length, int64 stride = 0) {
    std::cerr<<"Debugging Tmp memory of size:"<<length<<std::endl;
    int64 num_bytes = length * sizeof(double);

    double * tmp = (double*)malloc(num_bytes);
    cudaMemcpy(tmp, deviceMem, num_bytes, cudaMemcpyDeviceToHost);

    for(int64 i = 0; i< length; i++) {
        if(stride > 0 && i % stride == 0) {
            std::cerr<<std::endl;
        }
        //std::cerr<< std::fixed<<std::setprecision(2)<<tmp[i]<<" ";
        std::cerr<<std::setprecision(2);
        std::cerr<<tmp[i]<<" ";
    }
    std::cerr<<std::endl;
    delete [] tmp;
    std::cerr<<"Debugging Tmp memory Stop"<<std::endl;
}

void iDwt(MyVector & L, int levelsToReconstruct,
          int64 signalLength, int64 filterLength,
          double * coefficients,
          double * deviceLowReconstructFilter,
          double * deviceHighReconstructFilter,
          double * reconstructedSignal,
          double * extendedHighCoeff,
          double * extendedLowCoeff) {

    int currentCoefficientIndex = L.size() - 2 - 1;

    double * currentHighCoefficients;

    for(int i = 0; i < levelsToReconstruct; i++) {
        int64 currentSignalLength = L[currentCoefficientIndex + 1] - L[currentCoefficientIndex];

        int64 currentExtendedCoefficientLenght =
            currentSignalLength + (filterLength / 2) * 2;

        int threads;
        dim3 blocks;
        calculateBlockSize(currentExtendedCoefficientLenght, threads, blocks);

        int64 coefficientOffsetLow = L[currentCoefficientIndex];

        if(i == 0) {
            int64 coefficientOffsetHigh = L[currentCoefficientIndex + 1];
            currentHighCoefficients = coefficients + coefficientOffsetHigh;
        }

        extend<<<blocks, threads>>>(coefficients + coefficientOffsetLow, currentSignalLength,
                                    filterLength, extendedLowCoeff);

        extend<<<blocks, threads>>>(currentHighCoefficients, currentSignalLength,
                                    filterLength, extendedHighCoeff);

        //debugTmpMemory(extendedLowCoeff, currentExtendedCoefficientLenght);
        //debugTmpMemory(extendedHighCoeff, currentExtendedCoefficientLenght);

        calculateBlockSize(currentSignalLength * 2, threads, blocks);

        inverseConvolve<<<blocks, threads>>>(deviceLowReconstructFilter, deviceHighReconstructFilter,
                                             filterLength, extendedLowCoeff, extendedHighCoeff,
                                             reconstructedSignal, currentSignalLength * 2);
        currentCoefficientIndex--;
        currentHighCoefficients = reconstructedSignal;

    }
}

void dwt(MyVector & L, int levelsToCompress,
         double * deviceInputSignal, int64 signalLength,
         double * deviceLowFilter,
         double * deviceHighFilter,
         double * deviceOutputCoefficients,
         double * deviceLowCoefficientMemory,
         int64 filterLength) {

    int64 currentSignalLength = signalLength;
    int64 currentHighCoefficientOffset = 0 + signalLength / 2;
    //create a tempory low coefficient / signal extend array

    double * currentDeviceSignal = deviceInputSignal;

    for(int level = 0; level < levelsToCompress; level++) {

        //extend the signal
        int64 inputSignalExtendedLength = currentSignalLength + (filterLength / 2 ) * 2;

        int threads;
        dim3 blocks;
        calculateBlockSize(inputSignalExtendedLength, threads, blocks);

        extend<<<blocks, threads>>>(currentDeviceSignal, currentSignalLength,
                                    filterLength, deviceLowCoefficientMemory);

        //debugTmpMemory(deviceLowCoefficientMemory, inputSignalExtendedLength);
        ////convolve low filters
        int64 block_size = currentSignalLength / 2;

        int64 lowCoeffOffset = 0;
        if(level == levelsToCompress - 1) {
            lowCoeffOffset = L[level + 1] + signalLength / 2;
        }

        calculateBlockSize(block_size, threads, blocks);
        convolveWavelet<<<blocks, threads>>>(deviceLowFilter, filterLength,
                                             deviceLowCoefficientMemory, inputSignalExtendedLength,
                                             deviceOutputCoefficients, lowCoeffOffset);

        ////convolve high filters
        convolveWavelet<<<blocks, threads>>>(deviceHighFilter, filterLength,
                                             deviceLowCoefficientMemory, inputSignalExtendedLength,
                                             deviceOutputCoefficients, currentHighCoefficientOffset);


        currentSignalLength /= 2;
        currentHighCoefficientOffset = L[level + 1] + signalLength / 2;
        currentDeviceSignal = deviceOutputCoefficients;
    }
    //finally copy the low coefficients to the end
}
#endif
