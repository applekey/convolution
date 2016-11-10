#include "waveletFilter.h"
#include "helper.h"
#include <stdio.h>
#include <iostream>
#include <vector>

#define SIGNAL_LENGTH 16
#define COMPRESSION_LEVELS 5

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
    int signalLenght = SIGNAL_LENGTH;
    int extendedInputSignalLength = signalLenght + (9 - 1) * 2;

    double totalCoefficientLenght = 
        calculateCoefficientLength(coefficientIndicies, COMPRESSION_LEVELS, 
                               SIGNAL_LENGTH);

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

double * initOutput(int outputLength) {
    int num_bytes = outputLength * sizeof(double);
    cudaMalloc((void**)&device_output_array, num_bytes);
    return device_output_array;
}

double * initLowFilter() {
    int lowFilterLenght = 9;
    int num_bytes = lowFilterLenght * sizeof(double);

    host_low_filter_array = (double*)malloc(num_bytes);

    filter.getLowPassFilter(host_low_filter_array);

    for(int i =0; i < 9; i++) {
        printf("%f, ",host_low_filter_array[i]); 
    }
    cudaMalloc((void**)&device_low_filter_array, num_bytes);

    cudaMemcpy(device_low_filter_array, host_low_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_low_filter_array;
}

double * initHighFilter() {
    int highFilterLenght = 9;
    int num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array = (double*)malloc(num_bytes);

    filter.getHighPassFilter(host_high_filter_array);
    for(int i =0; i < 9; i++) {
        printf("%f, ",host_low_filter_array[i]); 
    }
    cudaMalloc((void**)&device_high_filter_array, num_bytes);

    cudaMemcpy(device_high_filter_array, host_high_filter_array, num_bytes, cudaMemcpyHostToDevice);
    return device_high_filter_array;
}

void transferMemoryBack(int outputLength) {
    int num_bytes = outputLength * sizeof(double);

    host_output_array = (double*)malloc(num_bytes);
    cudaMemcpy(host_output_array, device_output_array, num_bytes, cudaMemcpyDeviceToHost);  

    //print output
    printf("\n printing output \n");
    for(int i = 0; i < outputLength; i ++) {
        printf("%f \n", host_output_array[i]);
    }
}

void printOutputCoefficients(double * hostOutput, std::vector<int> coefficientIndicies) {
    int offset = SIGNAL_LENGTH / 2;
    int coefficientLevels = coefficientIndicies.size();
    
    for(int i = 0; i < coefficientLevels - 1;i++) {
        std::cerr<<"Level: "<<i<<std::endl;
        int levelCoefficientIndex = coefficientIndicies[i];
        int numberOfCoefficents = coefficientIndicies[i +1] - coefficientIndicies[i];

        for(int j = 0; j<numberOfCoefficents;j++) {
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


int main(int argc, const char * argv[]) {
    int outputLength = calculateCoefficientLength(coefficientIndicies, COMPRESSION_LEVELS, SIGNAL_LENGTH);
    outputLength += SIGNAL_LENGTH / 2; //add extra for buffer for first low coefficient

    filter.constructFilters();
    initLowFilter();
    initHighFilter();
    initSignal();
    initOutput(outputLength);

    //run filter   
    int levelsToCompress = 4;
    dwt(coefficientIndicies, levelsToCompress, 
        device_signal_array, SIGNAL_LENGTH,
        device_low_filter_array, device_high_filter_array,
        device_output_array, 9);

    //transfer output back
    transferMemoryBack(outputLength);
    //done free memory 
    freeMemory();

    return 0;
}
