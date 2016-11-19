#include "waveletFilter.h"
#include "helper.h"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cassert>

/*#define SIGNAL_LENGTH 134217728 */
/*#define SIGNAL_LENGTH 67108864 */
/*#define SIGNAL_LENGTH 1048576 */
#define SIGNAL_LENGTH 524288 
/*#define SIGNAL_LENGTH  32 */
#define COMPRESSION_LEVELS 3

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

//low reconstruct filters
double * host_low_reconstruct_filter_array = 0;
double * device_low_reconstruct_filter_array = 0;

//high reconstruct filters
double * host_high_reconstruct_filter_array = 0;
double * device_high_reconstruct_filter_array = 0;

//reconstructed signal
double * host_reconstruct_output_array = 0;
double * device_reconstruted_output_array = 0;

waveletFilter filter;

std::vector<int> coefficientIndicies; 

void initSignal() {

    long long num_bytes = SIGNAL_LENGTH * sizeof(double);
    assert(num_bytes != 0);

    host_signal_array = (double*)malloc(num_bytes);

    for(int i = 0; i < SIGNAL_LENGTH; i++) {
        /*host_signal_array[i] = 1.0 * sin((double)i /100.0) * 100.0;*/
        host_signal_array[i] = 1.0;
    }
}

void copyInputSignal() {

    long long num_bytes = SIGNAL_LENGTH * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&device_signal_array, num_bytes);

    if(err != cudaSuccess){
         printf("The error is %s", cudaGetErrorString(err));
    }
    cudaMemcpy(device_signal_array, host_signal_array, num_bytes, cudaMemcpyHostToDevice);
}

void initReconstructedSignal() {
    long long num_bytes = SIGNAL_LENGTH * sizeof(double);
    cudaError_t err = cudaMalloc((void**)&device_reconstruted_output_array, num_bytes);

    if(err != cudaSuccess){
         printf("The error is %s", cudaGetErrorString(err));
    }
}

void initOutput(int outputLength) {
    long long num_bytes = outputLength * sizeof(double);
    assert(num_bytes != 0);
    cudaError_t err = cudaMalloc((void**)&device_output_array, num_bytes);
    if(err != cudaSuccess){
         printf("The error is %s", cudaGetErrorString(err));
    }
}

void initLowFilter() {
    int lowFilterLenght = 9;
    long long num_bytes = lowFilterLenght * sizeof(double);

    host_low_filter_array = (double*)malloc(num_bytes);

    filter.getLowPassFilter(host_low_filter_array);

    cudaMalloc((void**)&device_low_filter_array, num_bytes);

    cudaMemcpy(device_low_filter_array, host_low_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void initHighFilter() {
    int highFilterLenght = 9;
    long long num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array = (double*)malloc(num_bytes);

    filter.getHighPassFilter(host_high_filter_array);
    cudaMalloc((void**)&device_high_filter_array, num_bytes);

    cudaMemcpy(device_high_filter_array, host_high_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void initLowReconstructFilter() {
    int lowFilterLenght = 9;
    long long num_bytes = lowFilterLenght * sizeof(double);

    host_low_reconstruct_filter_array = (double*)malloc(num_bytes);

    filter.getLowReconstructFilter(host_low_reconstruct_filter_array);
    cudaMalloc((void**)&device_low_reconstruct_filter_array, num_bytes);

    cudaMemcpy(device_low_reconstruct_filter_array, host_low_reconstruct_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void initHighReconstructFilter() {
    int highFilterLenght = 9;
    long long num_bytes = highFilterLenght * sizeof(double);

    host_high_reconstruct_filter_array = (double*)malloc(num_bytes);

    filter.getHighReconstructFilter(host_high_reconstruct_filter_array);
    cudaMalloc((void**)&device_high_reconstruct_filter_array, num_bytes);

    cudaMemcpy(device_high_reconstruct_filter_array, host_high_reconstruct_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void transferMemoryBack(int outputLength) {
    outputLength -=SIGNAL_LENGTH / 2; 
    long long num_bytes = outputLength * sizeof(double);
    assert(num_bytes != 0);

    host_output_array = (double*)malloc(num_bytes);
    cudaMemcpy(host_output_array, device_output_array + SIGNAL_LENGTH / 2, num_bytes, cudaMemcpyDeviceToHost);  
    /*cudaMemcpy(host_output_array, device_output_array, num_bytes, cudaMemcpyDeviceToHost);  */
}

void transferReconstructedMemoryBack(int outputLength) {
    long long num_bytes = outputLength * sizeof(double);
    assert(num_bytes != 0);

    host_reconstruct_output_array = (double*)malloc(num_bytes);
    cudaMemcpy(host_reconstruct_output_array, device_reconstruted_output_array, 
               num_bytes, cudaMemcpyDeviceToHost);  
}

void printOutputCoefficients(double * hostOutput, std::vector<int> coefficientIndicies) {
    int offset = 0;
    /*int offset = SIGNAL_LENGTH / 2;*/
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

void printReconstructedSignal() {
    std::cerr<<"Reconstructed Signal"<<std::endl;
    for(int i = 0 ; i< SIGNAL_LENGTH; i++) {
        std::cerr<<host_reconstruct_output_array[i]<<" ";
    } 
    std::cerr<<std::endl;
}
bool isCloseTo(double a, double b, double epsilon) {
    if(abs(a-b) < epsilon) {
        return true;
    } else {
        return false;
    }
}
void verifyReconstructedSignal() {
    bool allCorrect = true;
    std::cerr<<"Verifiying Signal"<<std::endl;
    for(int i = 0 ; i< SIGNAL_LENGTH; i++) {
        if(!isCloseTo(host_reconstruct_output_array[i],1, 0.0001)) {
         std::cerr<<host_reconstruct_output_array[i]<<std::endl;
          allCorrect = false;  
        }
    } 
    assert(allCorrect);
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

    free(host_low_reconstruct_filter_array);
    cudaFree(device_low_reconstruct_filter_array);

    free(host_high_reconstruct_filter_array);
    cudaFree(device_high_reconstruct_filter_array);
}

void writeResultsToMemory(double * output, int length) {
    /*double epsilon = 0.0000001;*/
    /*double a = -1.41442e-12;*/
    /*double b = 1.41421;*/

    /*int offset = SIGNAL_LENGTH / 2;*/
    /*for(int i = 0; i < length/2; i++) {*/
        /*if(abs(a -  output[i + offset]) < epsilon * 1.0e-12 ) {*/
            /*std::cerr<<"error "<<output[i + offset]<<std::endl;*/
        /*}*/
    /*}*/
    /*for(int i = length/2; i < length; i++) {*/
        /*if(abs(b -  output[i + offset]) < epsilon) {*/
            /*std::cerr<<"error "<<output[i + offset]<<std::endl;*/
        /*}*/
    /*}*/
    /*return;*/
    int offset = SIGNAL_LENGTH / 2;
    ofstream myfile;
    myfile.open("output.txt");
    
    for(int i = 0; i < length; i++) {
        myfile << output[i + offset]<<"\n";
    }
    myfile.close();
}


int main(int argc, const char * argv[]) {
    int outputLength = calculateCoefficientLength(coefficientIndicies, COMPRESSION_LEVELS, SIGNAL_LENGTH);
    outputLength += SIGNAL_LENGTH / 2; //add extra for buffer for first low coefficient

    filter.constructFilters();
    initLowFilter();
    initHighFilter();
    initLowReconstructFilter();
    initHighReconstructFilter();
    initSignal();
    initOutput(outputLength);
    initReconstructedSignal();

/*-------------------COMPRESS THE SIGNAL---------------------*/
auto startDecompose = std::chrono::system_clock::now();
    copyInputSignal();
    //run filter   
    dwt(coefficientIndicies, COMPRESSION_LEVELS, 
        device_signal_array, SIGNAL_LENGTH,
        device_low_filter_array, device_high_filter_array,
        device_output_array, 9);

    //transfer output back
    transferMemoryBack(outputLength);

auto endDecompose = std::chrono::system_clock::now();
std::chrono::duration<double> diff = endDecompose-startDecompose;
std::cout<< diff.count() << " s\n";
    /*printOutputCoefficients(host_output_array, coefficientIndicies);*/

    /*int ab = calculateCoefficientLength(coefficientIndicies, COMPRESSION_LEVELS, SIGNAL_LENGTH);*/
    /*writeResultsToMemory(host_output_array, ab);*/

/*-------------------DEBUG---------------------*/
/*std::cerr<<"low"<<std::endl;*/
/*for(int i = 0;i < 9; i++ ) {*/
    /*std::cerr<<host_low_reconstruct_filter_array[i]<<std::endl;*/
/*}*/

/*std::cerr<<"break"<<std::endl;*/

/*std::cerr<<"high"<<std::endl;*/
/*for(int i = 0;i < 9; i++ ) {*/
    /*std::cerr<<host_high_reconstruct_filter_array[i]<<std::endl;*/
/*}*/

/*-------------------UN-COMPRESS THE SIGNAL---------------------*/
auto startReconstruct = std::chrono::system_clock::now();
    iDwt(coefficientIndicies, COMPRESSION_LEVELS, 
         SIGNAL_LENGTH, 9, device_output_array + SIGNAL_LENGTH / 2,
         device_low_reconstruct_filter_array,
         device_high_reconstruct_filter_array,
         device_reconstruted_output_array);

    transferReconstructedMemoryBack(SIGNAL_LENGTH);
auto endReconstruct = std::chrono::system_clock::now();
diff = endReconstruct-startReconstruct;
std::cout<< diff.count() << " s\n";
    /*printReconstructedSignal();*/
    verifyReconstructedSignal();
    /*printReconstructedSignal();*/

/*-------------------CLEAN-UP---------------------*/
    //done free memory 
    freeMemory();

    return 0;
}
