#define SIGNAL_PAD_VALUE 0.0
#include <vector>
int calculateCoefficientLength(std::vector<int> &L, int levels,
                                int inputSignalLength) {

    int totalLength = 0;
    int currentCoefficientLength = inputSignalLength/2; //assume that all signals are powers of 2
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
    int inputIndex = index * 2 + (filterLength - 1); 

    double sum = 0.0;

    for(int i = 0; i < filterLength; i++) {
        sum += filter[i] * inputSignal[inputIndex - (filterLength - 1) + i];
    }

    output[index + outputOffset] = sum; 
}

__global__ void extend(double * inputSignal, int signalLength, int filterLength,
                       double * extendedSignal) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sideWidth = filterLength / 2;

    if(index <= sideWidth) {

        extendedSignal[index] = SIGNAL_PAD_VALUE;

    } else if(index < sideWidth + signalLength) {

        extendedSignal[index] = inputSignal[index - sideWidth];

    } else if(index > (sideWidth + signalLength) && (index <= signalLength + sideWidth * 2)) {

        extendedSignal[index] = SIGNAL_PAD_VALUE;

    } else {

        return;
    } 
}

double * initLowCoefficientMemory(int signalLength) {
    double * lowCoefficientMemory = 0; 
    int num_bytes = signalLength * sizeof(double);
    cudaMalloc((void**)&lowCoefficientMemory, num_bytes);
    return lowCoefficientMemory;
}

void dwt(std::vector<int> & L, int levelsToCompress,
         double * deviceInputSignal, int signalLength,
         double * deviceLowFilter, 
         double * deviceHighFilter,
         double * deviceOutputCoefficients,
         int outputLength,
         int filterLength) {

    int gridSize = 1;
    int currentSignalLength = signalLength;
    int currentHighCoefficientOffset = 0; 
    //create a tempory low coefficient / signal extend array
     
    int inputSignalExtendedLength = currentSignalLength + (9 - 1) * 2;
    double * deviceLowCoefficientMemory = initLowCoefficientMemory(inputSignalExtendedLength);

    for(int level = 0; level < levelsToCompress; level++) {

        //convolve high filters
        int outputOffset = 0;

        //extend the signal
        int inputSignalExtendedLength = currentSignalLength + (9 - 1) * 2;

        //extend<<<gridSize, block_size>>>(deviceHighFilter, 9, 
                        //deviceInputSignal, inputSignalExtendedLength,
                        //deviceOutputCoefficients, 0);

        //extend the signalLength
        int block_size = signalLength / 2;

        convolveWavelet<<<gridSize, block_size>>>(deviceHighFilter, 9, 
                        deviceInputSignal, inputSignalExtendedLength,
                        deviceOutputCoefficients, 0);
        
        ////convolve low filters
        convolveWavelet<<<gridSize, block_size>>>(deviceLowFilter, 9, 
                        device_signal_array, inputSignalExtendedLength,
                        device_output_array, currentHighCoefficientOffset);

        currentSignalLength /= 2;
        currentHighCoefficientOffset = L[level + 1] - L[level];
    }
}
