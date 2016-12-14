#include "waveletFilter.h"
#include "helper.h"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cassert>
#include <unistd.h>
#include "test2D.h"

/*--------------------------------------*/
// All the code below is to test 1D signal,
// the code to test 2D is in test2D.h
/*--------------------------------------*/
using namespace std;

int64 SIGNAL_LENGTH = 0;
int64 COMPRESSION_LEVELS = 0;
int64 PRINT_INTERMEDIATE = 0;

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

void initSignal() {

    int64 num_bytes = SIGNAL_LENGTH * sizeof(double);
    assert(num_bytes != 0);

    host_signal_array = (double *)malloc(num_bytes);

    for (int64 i = 0; i < SIGNAL_LENGTH; i++) {
        /*host_signal_array[i] = 1.0 * sin((double)i /100.0) * 100.0;*/
        /*host_signal_array[i] = 0.1 * float(i);*/
        host_signal_array[i] = 1.0;
    }
}

void copyInputSignal() {

    int64 num_bytes = SIGNAL_LENGTH * sizeof(double);
    cudaError_t err = cudaMalloc((void **)&device_signal_array, num_bytes);

    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
    cudaMemcpy(device_signal_array, host_signal_array, num_bytes, cudaMemcpyHostToDevice);
}

void initReconstructedSignal() {
    int64 num_bytes = SIGNAL_LENGTH * sizeof(double);
    cudaError_t err = cudaMalloc((void **)&device_reconstruted_output_array, num_bytes);

    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
}

void initOutput(int64 outputLength) {
    int64 num_bytes = outputLength * sizeof(double);
    assert(num_bytes != 0);
    cudaError_t err = cudaMalloc((void **)&device_output_array, num_bytes);
    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
}

void initLowFilter() {
    int64 lowFilterLenght = 9;
    int64 num_bytes = lowFilterLenght * sizeof(double);

    host_low_filter_array = (double *)malloc(num_bytes);

    filter.getLowPassFilter(host_low_filter_array);

    cudaMalloc((void **)&device_low_filter_array, num_bytes);

    cudaMemcpy(device_low_filter_array, host_low_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void initHighFilter() {
    int64 highFilterLenght = 9;
    int64 num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array = (double *)malloc(num_bytes);

    filter.getHighPassFilter(host_high_filter_array);
    cudaMalloc((void **)&device_high_filter_array, num_bytes);

    cudaMemcpy(device_high_filter_array, host_high_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void initLowReconstructFilter() {
    int64 lowFilterLenght = 9;
    int64 num_bytes = lowFilterLenght * sizeof(double);

    host_low_reconstruct_filter_array = (double *)malloc(num_bytes);

    filter.getLowReconstructFilter(host_low_reconstruct_filter_array);
    cudaMalloc((void **)&device_low_reconstruct_filter_array, num_bytes);

    cudaMemcpy(device_low_reconstruct_filter_array, host_low_reconstruct_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void initHighReconstructFilter() {
    int64 highFilterLenght = 9;
    int64 num_bytes = highFilterLenght * sizeof(double);

    host_high_reconstruct_filter_array = (double *)malloc(num_bytes);

    filter.getHighReconstructFilter(host_high_reconstruct_filter_array);
    cudaMalloc((void **)&device_high_reconstruct_filter_array, num_bytes);

    cudaMemcpy(device_high_reconstruct_filter_array, host_high_reconstruct_filter_array, num_bytes, cudaMemcpyHostToDevice);
}

void transferMemoryBack(int64 outputLength) {
    outputLength -= SIGNAL_LENGTH / 2;
    int64 num_bytes = outputLength * sizeof(double);
    assert(num_bytes != 0);

    /*cudaHostAlloc((void**)&host_output_array, num_bytes, */
    /*cudaHostAllocDefault) ;*/

    host_output_array = (double *)malloc(num_bytes);
    cudaMemcpy(host_output_array, device_output_array + SIGNAL_LENGTH / 2, num_bytes, cudaMemcpyDeviceToHost);
}

void transferReconstructedMemoryBack(int64 outputLength) {
    int64 num_bytes = outputLength * sizeof(double);
    assert(num_bytes != 0);

    host_reconstruct_output_array = (double *)malloc(num_bytes);
    cudaMemcpy(host_reconstruct_output_array, device_reconstruted_output_array,
               num_bytes, cudaMemcpyDeviceToHost);
}

void printOutputCoefficients(double * hostOutput, MyVector & coefficientIndicies) {
    int64 offset = 0;
    /*int offset = SIGNAL_LENGTH / 2;*/
    int coefficientLevels = coefficientIndicies.size();

    /*int total = coefficientIndicies[3];*/
    /*std::cerr<<coefficientLevels<<" "<<total<<std::endl;*/
    /*for(int i =0; i< total; i++) {*/
    /*std::cerr<<hostOutput[offset + i]<<std::endl;*/
    /*}*/

    for (int i = 0; i < coefficientLevels - 1; i++) {
        std::cerr << "Level: " << i << std::endl;
        int64 levelCoefficientIndex = coefficientIndicies[i];
        int64 numberOfCoefficents = coefficientIndicies[i + 1] - coefficientIndicies[i];

        for (int64 j = 0; j < numberOfCoefficents; j++) {
            double coeffVal = hostOutput[levelCoefficientIndex + j + offset];
            std::cerr << coeffVal << " ";
        }
        std::cerr << std::endl;
    }
}

void printReconstructedSignal() {
    std::cerr << "Reconstructed Signal" << std::endl;
    for (int64 i = 0 ; i < SIGNAL_LENGTH; i++) {
        std::cerr << host_reconstruct_output_array[i] << " ";
    }
    std::cerr << std::endl;
}
bool isCloseTo(double a, double b, double epsilon) {
    if (abs(a - b) < epsilon) {
        return true;
    } else {
        return false;
    }
}
void verifyReconstructedSignal() {
    bool allCorrect = true;
    std::cerr << "Verifiying Signal" << std::endl;
    for (int64 i = 0 ; i < SIGNAL_LENGTH; i++) {
        if (!isCloseTo(host_reconstruct_output_array[i], 1, 0.01)) {
            allCorrect = false;
        }
    }

    if(allCorrect) {
        std::cerr<<"all correct 1D"<<std::endl;
    } else {
        std::cerr<<"reconstruction error 1D"<<std::endl;
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

    free(host_low_reconstruct_filter_array);
    cudaFree(device_low_reconstruct_filter_array);

    free(host_high_reconstruct_filter_array);
    cudaFree(device_high_reconstruct_filter_array);
}

void writeResultsToMemory(double * output, int64 length) {
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
    int64 offset = SIGNAL_LENGTH / 2;
    ofstream myfile;
    myfile.open("output.txt");

    for (int64 i = 0; i < length; i++) {
        myfile << output[i + offset] << "\n";
    }
    myfile.close();
}

double * initTmpCoefficientMemory(int64 signalLength) {
    double * lowCoefficientMemory = 0;
    int64 num_bytes = signalLength * sizeof(double);
    assert(num_bytes != 0);
    cudaMalloc((void **)&lowCoefficientMemory, num_bytes);
    return lowCoefficientMemory;
}

void test1D() {
    std::cerr << "Testing 1D Decompose" << std::endl;
    MyVector coefficientIndicies;

    int64 outputLength = calculateCoefficientLength(coefficientIndicies, COMPRESSION_LEVELS, SIGNAL_LENGTH);
    outputLength += SIGNAL_LENGTH / 2; //add extra for buffer for first low coefficient

    filter.constructFilters();
    initLowFilter();
    initHighFilter();
    initLowReconstructFilter();
    initHighReconstructFilter();
    initSignal();
    initOutput(outputLength);
    initReconstructedSignal();

    int64 extendedSignalLength = SIGNAL_LENGTH + (SIGNAL_LENGTH / 2 ) * 2;
    double * tmpMemoryDWT = initTmpCoefficientMemory(extendedSignalLength);

    copyInputSignal();
    auto startDecompose = std::chrono::system_clock::now();
    /*-------------------COMPRESS THE SIGNAL---------------------*/
    //run filter
    dwt(coefficientIndicies, COMPRESSION_LEVELS,
        device_signal_array, SIGNAL_LENGTH,
        device_low_filter_array, device_high_filter_array,
        device_output_array, tmpMemoryDWT, 9);

    //transfer output back

    cudaDeviceSynchronize();
    auto endDecompose = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = endDecompose - startDecompose;
    std::cout << diff.count() << "  1D Compression Total s\n";
    transferMemoryBack(outputLength);

    cudaFree(tmpMemoryDWT);

    if(PRINT_INTERMEDIATE) {
        printOutputCoefficients(host_output_array, coefficientIndicies);
    };

    /*-------------------UN-COMPRESS THE SIGNAL---------------------*/
    double * tmpMemoryDWTHigh = initTmpCoefficientMemory(SIGNAL_LENGTH);
    double * tmpMemoryDWTLow = initTmpCoefficientMemory(SIGNAL_LENGTH);

    auto startReconstruct = std::chrono::system_clock::now();
    iDwt(coefficientIndicies, COMPRESSION_LEVELS,
         SIGNAL_LENGTH, 9, device_output_array + SIGNAL_LENGTH / 2,
         device_low_reconstruct_filter_array,
         device_high_reconstruct_filter_array,
         device_reconstruted_output_array,
         tmpMemoryDWTHigh, tmpMemoryDWTLow);
    cudaDeviceSynchronize();
    auto endReconstruct = std::chrono::system_clock::now();

    diff = endReconstruct - startReconstruct;
    std::cout << diff.count() << "  1D De-Compression Total s\n";
    transferReconstructedMemoryBack(SIGNAL_LENGTH);
    verifyReconstructedSignal();

    if(PRINT_INTERMEDIATE) {
        printReconstructedSignal();
    };

    /*-------------------CLEAN-UP---------------------*/
    //done free memory
    cudaFree(tmpMemoryDWTHigh);
    cudaFree(tmpMemoryDWTLow);
    freeMemory();
}

void verifyTimer() {
    std::cerr<<"Testing timer, should be 1 second"<<std::endl;
    auto start = std::chrono::system_clock::now();
    usleep(1000000);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "timer reported: "<<diff.count() << " s\n";
}

int isPowerOfTwo (unsigned int x)
{
 while (((x % 2) == 0) && x > 1) /* While x is even and > 1 */
   x /= 2;
 return (x == 1);
}

int main(int argc, const char * argv[]) {
    //scrub input
    verifyTimer();

    if(argc <= 4) {
        std::cerr<<"incorrect args, example is ./wave 16384 3 1, args are signal size, level of compression, test number"<<std::endl;
        return 0;
    }

    int N = atoi(argv[1]);
    int levels = atoi(argv[2]);
    int test = atoi(argv[3]);
    int printIntermediate = atoi(argv[4]);
    assert(N > 0);
    assert(levels > 0);
    assert(test > 0);
    assert(printIntermediate == 0 ||  printIntermediate == 1);

#if defined SHARED_MEMORY
    std::cerr<<"Running with shared memory optimization"<<std::endl;
#endif
#if defined BIG 
    std::cerr<<"Running large sizes >= 1024"<<std::endl;
#endif

    if(!isPowerOfTwo(N)) {
        std::cerr<<"N,"<<N<<" is not a power of 2"<<std::endl;
        return 0;
    }

    std::cerr<<"N is: "<<N<<" levels is: "<<levels<<std::endl;
    SIGNAL_LENGTH = N;
    COMPRESSION_LEVELS = levels;
    PRINT_INTERMEDIATE = printIntermediate;
    
    if(test == 1) {
        test1D();
    } else if(test == 2) {
        test2D(SIGNAL_LENGTH, COMPRESSION_LEVELS);
    }
    return 0;
}
