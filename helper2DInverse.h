#include "defines.h"
#define OFFSETVAL 2
#define HIGH_LEFT 1
#define HIGH_RIGHT 0
#define LOW_LEFT 0
#define LOW_RIGHT 1

struct vec2 {
    vec2(int x, int y) {
        x = x;
        y = y;
    }
    int64 x, y;
};

struct ImageMeta {
    int64 imageWidth, imageHeight;
    int64 xStart, xEnd;
    int64 yStart, yEnd;
};

/*------------------------------------------------------HORIZONTAL------------------------------*/
__global__ void inverseConvolveHorizontal(double * inputSignal, int64 filterLength,
        int64 totalSignalLength,
        double * lowFilter, double * highFilter,
        struct ImageMeta inputImageMeta,
        double * reconstructedSignal,
        char * mirrorIndexLow, char * mirrorIndexHigh, int mirrorLength) {
    int64 index = calculateIndex();

    if (index >= totalSignalLength) {
        return;
    }

    int64 stride = inputImageMeta.imageWidth;

    int64 blockWidth = inputImageMeta.xEnd;
    int64 yIndexLocal = index / blockWidth;
    int64 xIndexLocal = index % blockWidth;

    int64 yLowStart, yHighStart;

    if(xIndexLocal % 2 == 0) {
        yLowStart = filterLength - 1;
        yHighStart = filterLength - 2;
    } else {
        yLowStart = filterLength - 2;
        yHighStart = filterLength - 1;
    }

    int64 filterSideWidth = filterLength / 2;

    // do extension here
    double sum = 0;

    //populate vals
    int64 highCoefficientOffsetX = blockWidth / 2;

#if defined SHARED_MEMORY
    __shared__ double sLowfilter[9]; //max per
    __shared__ double sHighfilter[9]; //max per
    if(int64(threadIdx.x) < int64(9)) {
        sLowfilter[threadIdx.x] = lowFilter[threadIdx.x];
        sHighfilter[threadIdx.x] = highFilter[threadIdx.x];
    }

    __syncthreads();
#endif

    //low
    double valsLow[9];
    int64 lowCoefficientIndex = (xIndexLocal + 1) / 2;

    for (int i = 0; i < 9; i++) {
        if(lowCoefficientIndex + i < 4 || lowCoefficientIndex + i >= blockWidth / 2 + 4) {
            int64 indexOffset = mirrorIndexLow[lowCoefficientIndex + i];
            valsLow[i] = inputSignal[yIndexLocal * stride + (lowCoefficientIndex - filterSideWidth + i) + indexOffset];
        } else {
            valsLow[i] = inputSignal[yIndexLocal * stride + (lowCoefficientIndex - filterSideWidth + i)];
        }
    }

    int64 offsetVals = OFFSETVAL;
    int64 iC = 0;
    while(yLowStart > -1) {
#if defined SHARED_MEMORY
        sum += sLowfilter[yLowStart] * valsLow[iC + offsetVals];
#else
        sum += lowFilter[yLowStart] * valsLow[iC + offsetVals];
#endif
        yLowStart -= 2;
        iC++;
    }

    //high
    double valsHigh[9];
    int64 highCoefficientIndex = xIndexLocal / 2;

    for (int i = 0; i < 9; i++) {
        int64 xComponent = 0;
        if(highCoefficientIndex + i < 4 || highCoefficientIndex + i >= blockWidth / 2 + 4) {
            int64 indexOffset = mirrorIndexHigh[highCoefficientIndex + i];
            xComponent = (highCoefficientIndex - filterSideWidth + highCoefficientOffsetX + i + indexOffset);
        } else {
            xComponent = (highCoefficientIndex - filterSideWidth + highCoefficientOffsetX + i);
        }
        valsHigh[i] = inputSignal[yIndexLocal * stride + xComponent];
    }

    iC = 0;
    while(yHighStart > -1) {
#if defined SHARED_MEMORY
        sum += sHighfilter[yHighStart] * valsHigh[iC + offsetVals];
#else
        sum += highFilter[yHighStart] * valsHigh[iC+ offsetVals];
#endif
        iC++;
        yHighStart -= 2;
    }

    int64 outputIndex = yIndexLocal * stride + xIndexLocal;
    reconstructedSignal[outputIndex] = sum;
}

/*------------------------------------------------------VERTICAL------------------------------*/
__global__ void inverseConvolveVertical(double * inputSignal, int64 filterLength,
                                        int64 totalSignalLength,
                                        double * lowFilter, double * highFilter,
                                        struct ImageMeta inputImageMeta,
                                        double * reconstructedSignal,
                                        int64 maxThreadWidth,
                                        char * mirrorIndexLow, char * mirrorIndexHigh) {
    int64 index = calculateIndex();

    if (index >= totalSignalLength) {
        return;
    }

    int64 stride = inputImageMeta.imageWidth;
    int64 blockWidth = inputImageMeta.xEnd;
    int64 blockHeight = inputImageMeta.yEnd;

    int64 yRoll =  index / maxThreadWidth;
    int64 yIndexLocal = yRoll % blockHeight;
    int64 yRollX =  yRoll / blockHeight;
    int64 xIndexLocal = (index % maxThreadWidth) + maxThreadWidth * yRollX;

    //int64 yIndexLocal = index / blockWidth;
    //int64 xIndexLocal = index % blockWidth;

    int64 filterSideWidth = filterLength / 2;

    int64 yLowStart, yHighStart;

    if(yIndexLocal % 2 == 0) {
        yLowStart = filterLength - 1;
        yHighStart = filterLength - 2;
    } else {
        yLowStart = filterLength - 2;
        yHighStart = filterLength - 1;
    }

#if defined SHARED_MEMORY
    __shared__ double sLowfilter[9]; //max per
    __shared__ double sHighfilter[9]; //max per
    if(int64(threadIdx.x) < int64(9)) {
        sLowfilter[threadIdx.x] = lowFilter[threadIdx.x];
        sHighfilter[threadIdx.x] = highFilter[threadIdx.x];
    }
    __syncthreads();
#endif

    // do extension here
    double sum = 0;

    //populate vals
    int64 highCoefficientOffsetY = blockWidth / 2;

    //low
    double valsLow[9];
    int64 lowCoefficientIndex = (yIndexLocal + 1) / 2;

    int fillLeft = filterSideWidth - lowCoefficientIndex;
    int filledL = 0;

    for (int i = 0; i < fillLeft; i++) {
        int64 mirrorDistance = fillLeft - i - LOW_LEFT;
        valsLow[i] = inputSignal[mirrorDistance * stride +  xIndexLocal];
        filledL += 1;
    }

    int fillRight = lowCoefficientIndex - (highCoefficientOffsetY - filterSideWidth - 1);
    int filledR = 0;
    for (int i = 0; i < fillRight; i++) {
        int64 mirrorDistance = fillRight - i - LOW_RIGHT;
        valsLow[8 - i] = inputSignal[(highCoefficientOffsetY - 1 - mirrorDistance) * stride
                                     + xIndexLocal ];
        filledR += 1;
    }

    for (int i = filledL; i < 9 - filledR; i++) {
        valsLow[i] = inputSignal[(lowCoefficientIndex - filterSideWidth + i) * stride
                                 + xIndexLocal ];
    }

    int64 offsetVals = OFFSETVAL;
    int64 iC = 0;
    while(yLowStart > -1) {
#if defined SHARED_MEMORY
        sum += sLowfilter[yLowStart] * valsLow[iC + offsetVals];
#else
        sum += lowFilter[yLowStart] * valsLow[iC + offsetVals];
#endif
        yLowStart -= 2;
        iC++;
    }

    //high
    double valsHigh[9];
    int64 highCoefficientIndex = yIndexLocal / 2;
    fillLeft = filterSideWidth - highCoefficientIndex;
    filledL = 0;

    for (int i = 0; i < fillLeft; i++) {
        int64 mirrorDistance = fillLeft - i - HIGH_LEFT;
        valsHigh[i] = inputSignal[(highCoefficientOffsetY + mirrorDistance) * stride
                                  +  xIndexLocal ];
        filledL += 1;
    }

    fillRight = highCoefficientIndex - (highCoefficientOffsetY - filterSideWidth - 1);
    filledR = 0;
    for (int i = 0; i < fillRight; i++) {
        int64 mirrorDistance = fillRight - i - HIGH_RIGHT;
        valsHigh[8 - i] = inputSignal[(2 * highCoefficientOffsetY - 1 - mirrorDistance) * stride
                                      + xIndexLocal ];
        filledR += 1;
    }

    for (int i = filledL; i < 9 - filledR; i++) {
        valsHigh[i] = inputSignal[(highCoefficientIndex + highCoefficientOffsetY - filterSideWidth + i) * stride
                                  + xIndexLocal ];
    }

    iC = 0;
    while(yHighStart > -1) {
#if defined SHARED_MEMORY
        sum += sHighfilter[yHighStart] * valsHigh[iC + offsetVals];
#else
        sum += highFilter[yHighStart] * valsHigh[iC + offsetVals];
#endif
        yHighStart -= 2;
        iC ++;
    }

    int64 outputIndex = yIndexLocal * stride + xIndexLocal;
    reconstructedSignal[outputIndex] = sum;
}


/*------------------------------------------------------CALLER------------------------------*/
void iDwt2D(int levelsToCompressUncompress,
            double * deviceInputSignal,
            struct ImageMeta & inputImageMeta,
            double * deviceILowFilter,
            double * deviceIHighFilter,
            int64 filterLength,
            double * deviceOutputCoefficients,
            double * deviceTmpMemory,
            char * deviceIndexLow,
            char * deviceIndexHigh,
            int mirrorLength) {

    bool isHorizontal = false;
    //calculate current image meta
    struct ImageMeta currentImageMeta = inputImageMeta;

    for (int level = 0; level < levelsToCompressUncompress * 2; level++) {

        auto startLocal = std::chrono::system_clock::now();

        if (isHorizontal) {
            int64 totalNumElements = currentImageMeta.xEnd *  currentImageMeta.yEnd;
            int threads;
            dim3 blocks;
            calculateBlockSize(totalNumElements, threads, blocks);

            inverseConvolveHorizontal <<< blocks, threads>>>(deviceTmpMemory, filterLength,
                    totalNumElements,
                    deviceILowFilter, deviceIHighFilter,
                    currentImageMeta,
                    deviceOutputCoefficients, deviceIndexLow, deviceIndexHigh, mirrorLength);
        } else {
            //expand current image size

            if(level != 0) {
                cudaMemset(deviceIndexLow, 0, 4 * sizeof(char));
                cudaMemset(deviceIndexHigh, 0, 4 * sizeof(char));
                deviceIndexLow -= (currentImageMeta.yEnd / 2);
                deviceIndexHigh -= (currentImageMeta.yEnd / 2);
            }

            currentImageMeta.xEnd *= 2;
            currentImageMeta.yEnd *= 2;


            int64 totalNumElements = currentImageMeta.xEnd *  currentImageMeta.yEnd;
            int threads;
            dim3 blocks;
            calculateBlockSize(totalNumElements, threads, blocks);

            int64 maxThreadWidth = currentImageMeta.xEnd;

            if(maxThreadWidth > MAX_SIDE) {
                maxThreadWidth = MAX_SIDE;
            }

            inverseConvolveVertical <<< blocks, threads>>>(deviceInputSignal, filterLength,
                    totalNumElements,
                    deviceILowFilter, deviceIHighFilter,
                    currentImageMeta,
                    deviceTmpMemory,
                    maxThreadWidth, deviceIndexLow, deviceIndexHigh);
        }
        //cudaDeviceSynchronize();

        auto endLocal = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endLocal - startLocal;
        //std::cerr<<"DWT-I level: "<<level/2<<" size: "<<currentImageMeta.xEnd<<", calc: "<<diff.count()<<std::endl;

        isHorizontal = !isHorizontal;
    }
}
