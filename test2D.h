#include "waveletFilter.h"
#include "helper2D.h"
#include <cstring>

int64 SIGNAL_LENGTH_2D = 0;
int64 COMPRESSION_LEVELS_2D = 0;

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

struct signalGenerator2D {
    double valueGivenIndex(int64 index, int64 maxIndex) {
        /*return 1.0;*/
        return 0.1 * double(index);
    }

    double calculateRMSE(double * reconstructedSignal, int64 maxIndex) {
        double errorSum = 0;
        for(int64 i = 0; i < maxIndex; i++ ) {
            double differenceSqured = (reconstructedSignal[i] - valueGivenIndex(i, maxIndex)) *
                                      (reconstructedSignal[i] - valueGivenIndex(i, maxIndex));
            errorSum += differenceSqured;
        }
        return sqrt(errorSum / float(maxIndex));
    }
};

struct signalGenerator2D sigGenerator2D;

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

    host_signal_array_2D = (double *)malloc(num_bytes);

    for (int64 i = 0; i < signalLength; i++) {
        host_signal_array_2D[i] = sigGenerator2D.valueGivenIndex(i, signalLength);
    }
}

void initReconstructedSignal2D() {

    int64 signalLength = get1DSignalLength();
    int64 num_bytes = signalLength * sizeof(double);
    assert(num_bytes != 0);

    host_reconstructed_signal_array_2D = (double *)malloc(num_bytes);

    cudaError_t err = cudaMalloc((void **)&device_reconstructed_signal_array_2D, num_bytes);
}

void copyInputSignal2D() {

    int64 num_bytes = get1DSignalLength() * sizeof(double);
    cudaError_t err = cudaMalloc((void **)&device_signal_array_2D, num_bytes);

    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
    cudaMemcpy(device_signal_array_2D, host_signal_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initLowFilter_2D() {
    int64 lowFilterLenght = 9;
    int64 num_bytes = lowFilterLenght * sizeof(double);

    host_low_filter_array_2D = (double *)malloc(num_bytes);

    filter2D.getLowPassFilter(host_low_filter_array_2D);

    cudaMalloc((void **)&device_low_filter_array_2D, num_bytes);

    cudaMemcpy(device_low_filter_array_2D, host_low_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initHighFilter_2D() {
    int64 highFilterLenght = 9;
    int64 num_bytes = highFilterLenght * sizeof(double);

    host_high_filter_array_2D = (double *)malloc(num_bytes);

    filter2D.getHighPassFilter(host_high_filter_array_2D);
    cudaMalloc((void **)&device_high_filter_array_2D, num_bytes);

    cudaMemcpy(device_high_filter_array_2D, host_high_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initLowInverseFilter_2D() {
    int64 lowFilterInverseLenght = 9;
    int64 num_bytes = lowFilterInverseLenght * sizeof(double);

    host_low_inverse_filter_array_2D = (double *)malloc(num_bytes);

    filter2D.getLowReconstructFilter(host_low_inverse_filter_array_2D);

    cudaMalloc((void **)&device_low_inverse_filter_array_2D, num_bytes);

    cudaMemcpy(device_low_inverse_filter_array_2D, host_low_inverse_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initHighInverseFilter_2D() {
    int64 highInverseFilterLenght = 9;
    int64 num_bytes = highInverseFilterLenght * sizeof(double);

    host_high_inverse_filter_array_2D = (double *)malloc(num_bytes);

    filter2D.getHighReconstructFilter(host_high_inverse_filter_array_2D);
    cudaMalloc((void **)&device_high_inverse_filter_array_2D, num_bytes);

    cudaMemcpy(device_high_inverse_filter_array_2D, host_high_inverse_filter_array_2D, num_bytes, cudaMemcpyHostToDevice);
}

void initOutput_2D() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    cudaError_t err = cudaMalloc((void **)&device_output_array_2D, num_bytes);
    if (err != cudaSuccess) {
        printf("The error is %s", cudaGetErrorString(err));
    }
}

void initDeviceTmpMemory() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    assert(num_bytes != 0);
    cudaMalloc((void **)&deviceTmpMemory, num_bytes);
}


void transferMemoryBack_2D() {
    int64 num_bytes = get1DSignalLength() * sizeof(double);
    assert(num_bytes != 0);

    host_output_array_2D = (double *)malloc(num_bytes);
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

    for (int64 i = 0; i < signalLenght; i++) {
        if (i % stride == 0) {
            std::cerr << std::endl;
        }
        std::cerr << signal[i] << " ";
    }
    std::cerr << std::endl;
}


bool isCloseTo2D(double a, double b, double epsilon) {
    if (abs(a - b) < epsilon) {
        return true;
    } else {
        return false;
    }
}

void verifyReconstructedSignal2D() {
    int64 sigLength = get1DSignalLength();
    double rmse = sigGenerator2D.calculateRMSE(host_output_array_2D, sigLength);
    std::cerr<<"RMSE: "<<rmse<<std::endl;

    bool allCorrect = true;
    std::cerr << "Verifiying Signal 2D" << std::endl;

    for (int64 i = 0 ; i < sigLength; i++) {
        if (!isCloseTo2D(host_output_array_2D[i], sigGenerator2D.valueGivenIndex(i, sigLength), 0.01)) {
            allCorrect = false;
        }
    }

    if(allCorrect) {
        std::cerr<<"all correct 2D"<<std::endl;
    } else {
        std::cerr<<"reconstruction error 2D"<<std::endl;
    }
    return;
}

int * host_index;
int * device_index;
void generateIndexArray() {
    int64 indexLen = SIGNAL_LENGTH_2D + (9 / 2) * 2;
    int64 num_bytes = indexLen * sizeof(int);

    host_index = (int *)malloc(num_bytes);
    std::memset(host_index, 0, num_bytes);

    for(int64 i = 0; i < indexLen; i++) {
        if(i < 4) {
            host_index[i] = ((4 - i) * 2);
        }
        else if(i >= indexLen - 4) {
            host_index[i] = -((i - (indexLen -4) + 1) * 2);
        }
    }

    cudaMalloc((void **)&device_index, num_bytes);
    cudaMemcpy(device_index, host_index, num_bytes, cudaMemcpyHostToDevice);
}

char * host_index_inverse_low;
char * device_index_inverse_low;
void generateIndexArrayInverse(int compressionLevels, int signalLength) {
    int64 indexLen = signalLength + (9 / 2) * 2;
    int64 num_bytes = indexLen * sizeof(char);

    host_index_inverse_low = (char *)malloc(num_bytes);
    std::memset(host_index_inverse_low, 0, num_bytes);

    int cur = 0;
    int totalOffset = 0;
    for(int i = 0;i < compressionLevels;i++) {

        host_index_inverse_low[cur] = ((4 - 0) * 2);
        host_index_inverse_low[cur+ 1] = ((4 - 1) * 2);
        host_index_inverse_low[cur + 2] = ((4 - 2) * 2);
        host_index_inverse_low[cur + 3] = ((4 - 3) * 2);
        signalLength /= 2;
        cur += signalLength;
        if( i != compressionLevels - 1) {
            totalOffset += signalLength;
        }
    }

    host_index_inverse_low[indexLen - 1] = -((indexLen - 1 - (indexLen -4) + 1) * 2) + 1;
    host_index_inverse_low[indexLen - 2] = -((indexLen - 2 - (indexLen -4) + 1) * 2) + 1;
    host_index_inverse_low[indexLen - 3]= -((indexLen - 3 - (indexLen -4) + 1) * 2) + 1;
    host_index_inverse_low[indexLen - 4]= -((indexLen - 4 - (indexLen -4) + 1) * 2) + 1;

    cudaMalloc((void **)&device_index_inverse_low, num_bytes);
    cudaMemcpy(device_index_inverse_low, host_index_inverse_low, num_bytes, cudaMemcpyHostToDevice);
    device_index_inverse_low += totalOffset;
}

char * host_index_inverse_high;
char * device_index_inverse_high;
void generateIndexArrayInverseHigh(int compressionLevels, int signalLength) {
    int64 indexLen = signalLength + (9 / 2) * 2;
    int64 num_bytes = indexLen * sizeof(char);

    host_index_inverse_high = (char *)malloc(num_bytes);
    std::memset(host_index_inverse_high, 0, num_bytes);

    int cur = 0;
    int totalOffset = 0;
    for(int i = 0;i < compressionLevels;i++) {

        host_index_inverse_high[cur] = ((4 - 0) * 2) - 1;
        host_index_inverse_high[cur+ 1] = ((4 - 1) * 2) - 1;
        host_index_inverse_high[cur + 2] = ((4 - 2) * 2) - 1;
        host_index_inverse_high[cur + 3] = ((4 - 3) * 2) - 1;
        signalLength /= 2;
        cur += signalLength;
        if( i != compressionLevels - 1) {
            totalOffset += signalLength;
        }
    }

    host_index_inverse_high[indexLen - 1] = -((indexLen - 1 - (indexLen -4) + 1) * 2);
    host_index_inverse_high[indexLen - 2] = -((indexLen - 2 - (indexLen -4) + 1) * 2);
    host_index_inverse_high[indexLen - 3]= -((indexLen - 3 - (indexLen -4) + 1) * 2);
    host_index_inverse_high[indexLen - 4]= -((indexLen - 4 - (indexLen -4) + 1) * 2);

    cudaMalloc((void **)&device_index_inverse_high, num_bytes);
    cudaMemcpy(device_index_inverse_high, host_index_inverse_high, num_bytes, cudaMemcpyHostToDevice);
    device_index_inverse_high += totalOffset;
}

void test2D(int64 signalLength2D, int64 compressionLevels, int PRINT_INTERMEDIATE) {
    std::cerr << "Testing 2D Decompose" << std::endl;

    SIGNAL_LENGTH_2D = signalLength2D;
    COMPRESSION_LEVELS_2D = compressionLevels;

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
    struct ImageMeta imageMeta;
    imageMeta.imageWidth = SIGNAL_LENGTH_2D;
    imageMeta.imageHeight = SIGNAL_LENGTH_2D;
    imageMeta.xStart = 0;
    imageMeta.yStart = 0;
    imageMeta.xEnd = SIGNAL_LENGTH_2D;
    imageMeta.yEnd = SIGNAL_LENGTH_2D;


    auto startDecompose = std::chrono::system_clock::now();
    generateIndexArray();
    struct ImageMeta compressionResultMeta = dwt2D(COMPRESSION_LEVELS_2D, device_signal_array_2D,
            imageMeta, device_low_filter_array_2D,
            device_high_filter_array_2D, 9, imageMeta,
            device_output_array_2D, deviceTmpMemory, &device_index, &host_index);

    cudaDeviceSynchronize();
    auto endDecompose = std::chrono::system_clock::now();
    transferMemoryBack_2D();
    std::chrono::duration<double> diff = endDecompose - startDecompose;
    std::cerr<<std::endl;
    std::cout << diff.count() << " 2D Compression Total s\n";
    std::cerr<<std::endl;

    if(PRINT_INTERMEDIATE) {
        printResult_2D(host_output_array_2D);
    }

    generateIndexArrayInverse(COMPRESSION_LEVELS_2D, SIGNAL_LENGTH_2D / 2);
    generateIndexArrayInverseHigh(COMPRESSION_LEVELS_2D, SIGNAL_LENGTH_2D / 2);
    auto startRecompose = std::chrono::system_clock::now();
    iDwt2D(COMPRESSION_LEVELS_2D,
           device_output_array_2D,
           compressionResultMeta,
           device_low_inverse_filter_array_2D,
           device_high_inverse_filter_array_2D,
           9,
           device_output_array_2D,
           deviceTmpMemory,
           device_index_inverse_low,
           device_index_inverse_high,
           SIGNAL_LENGTH_2D / 2 );

    cudaDeviceSynchronize();
    auto endRecompose = std::chrono::system_clock::now();
    transferMemoryBack_2D();
    diff = endRecompose - startRecompose;

    std::cerr<<std::endl;
    std::cout << diff.count() << " 2D De-Compression Total s\n";
    std::cerr<<std::endl;

    if(PRINT_INTERMEDIATE) {
        printResult_2D(host_output_array_2D);
    }

    verifyReconstructedSignal2D();
}
