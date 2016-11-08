#include <iostream>
#include <vector>
#include <cuda.h>
#include "waveletCompressor.h"

void printCoefficients(std::vector<int>& coefficientLengths, double * outputSignal) {
    int coefficientLevels = coefficientLengths.size();
    //last coefficient level is just a place holder
    for(int i = 0; i < coefficientLevels - 1;i++) {
        std::cerr<<"Level: "<<i<<std::endl;
        int levelCoefficientIndex = coefficientLengths[i];
        int numberOfCoefficents = coefficientLengths[i +1] - coefficientLengths[i];
        for(int j = 0; j<numberOfCoefficents;j++) {
            double coeffVal = outputSignal[levelCoefficientIndex + j];
            std::cerr<<coeffVal<<" ";
        }
        std::cerr<<std::endl;
    }
}

int main(int argc, const char * argv[]) {
    waveletCompressor compressor;
    //generate constant signal that is power of 2, 64
    int signalLength = 64;
    double inputSignal[64];

    for(int i = 0; i < 64; i++) {
        inputSignal[0] = 1.0;
    }

    int levelsToCompress = 2;
    std::vector<int> coefficientLengths;
    /*double * coefficients = compressor.compressWave(inputSignal, signalLength, */
                                                        /*levelsToCompress, coefficientLengths);*/
    printCoefficients(coefficientLengths, coefficients);
    return 0;
}
