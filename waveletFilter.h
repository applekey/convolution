#include <iostream>
class waveletFilter {
private:
    int filterLength = 9;
    double * lowPassFilter;
    double * highPassFilter;
    
    double * lowReconstructFilter;
    double * highReconstructFilter;

public:
    void getLowPassFilter(double * copyTo) {
        memcpy(copyTo, lowPassFilter, filterLength * sizeof(double));
    }

    void getHighPassFilter(double * copyTo) {
        memcpy(copyTo, highPassFilter, filterLength * sizeof(double));
    }

    void getLowReconstructFilter(double * copyTo) {
        memcpy(copyTo, lowReconstructFilter, filterLength * sizeof(double));
    }

    void getHighReconstructFilter(double * copyTo) {
        memcpy(copyTo, highReconstructFilter, filterLength * sizeof(double));
    }

    int getFilterLength() {
        return filterLength;
    }

    void allocateFilterMemory() {
        //cuda alloc here
        lowPassFilter = new double[filterLength];
        highPassFilter = new double[filterLength];
    }

    void deallocFilterMemory() {
        //cuda dealloc here
        delete [] lowPassFilter;
        delete [] highPassFilter;
    }

    ~waveletFilter() {
        deallocFilterMemory();
    }

    void constructFilters() {
        allocateFilterMemory();
        reverseFilter(hm4_44, lowPassFilter);
        qmfReverseFilter(h4, highPassFilter);
        lowReconstructFilter = h4;
        qmfFilter(hm4_44, highReconstructFilter);
    }


    void reverseFilter(double * input, double * output) {
        for(int i = 0; i < filterLength; i++) {
            output[filterLength - 1 - i] = input[i];
        }
    }

    void qmfFilter(double * input, double * output) {
        double modifier = (filterLength % 2 == 0) ? 1 : -1;

        for (int index = 0; index < filterLength; index++) {
            output[index] = modifier * input[filterLength - index - 1];
            modifier = -modifier;
        }
    }

    // flip filter at center
    void qmfReverseFilter(double * input, double * output) {
        double * tmpOutput = new double[filterLength];
        qmfFilter(input, tmpOutput);
        for(int i = 0; i < filterLength / 2; i++) {
            double front = tmpOutput[i];
            double back = tmpOutput[filterLength - 1 - i];
            output[i] = back;
            output[filterLength - 1 - i] = front;
        }
        int midIndex =  filterLength/2;
        output[midIndex] = tmpOutput[midIndex];
        delete [] tmpOutput;
    }


    double hm4_44[9] = {
        0.03782845550726404,
        -0.023849465019556843,
        -0.11062440441843718,
        0.37740285561283066,
        0.85269867900889385,
        0.37740285561283066,
        -0.11062440441843718,
        -0.023849465019556843,
        0.03782845550726404
    };

    double h4[9] = {
        0.0,
        -0.064538882628697058,
        -0.040689417609164058,
        0.41809227322161724,
        0.7884856164055829,
        0.41809227322161724,
        -0.040689417609164058,
        -0.064538882628697058,
        0.0
    };
};
