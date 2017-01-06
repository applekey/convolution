#include "defines.h"
#include "helper.h"

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
        char * mirrorIndexLow, char * mirrorIndexHigh) {
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

    int64 lowCoefficientIndex = (xIndexLocal + 1) / 2;
    int64 highCoefficientIndex = xIndexLocal / 2;

#if defined SHARED_MEMORY
    __shared__ double sLowfilter[9]; //max per
    __shared__ double sHighfilter[9]; //max per
    if(int64(threadIdx.x) < int64(9)) {
        sLowfilter[threadIdx.x] = lowFilter[threadIdx.x];
        sHighfilter[threadIdx.x] = highFilter[threadIdx.x];
    }

    __shared__ double lowVals[1024 + 8];
    __shared__ double highVals[1024 + 8];

    int64 localLowIndex =  (threadIdx.x + 1)/ 2;
    int64 localHighIndex =  highCoefficientIndex % 512;

    //low
    int64 indexOffset = (int64) mirrorIndexLow[lowCoefficientIndex];
    lowVals[localLowIndex] = inputSignal[yIndexLocal * stride + (lowCoefficientIndex - filterSideWidth ) + indexOffset];

    //high
    indexOffset = (int64) mirrorIndexHigh[highCoefficientIndex];
    int64 xOfff = (highCoefficientIndex - filterSideWidth + highCoefficientOffsetX + indexOffset);

    highVals[localHighIndex] = inputSignal[yIndexLocal * stride + xOfff];

    if(threadIdx.x == 1023) {
        for(int i = 0; i < 8; i++) {
            //low
            indexOffset = (int64) mirrorIndexLow[lowCoefficientIndex + 1 + i];
            lowVals[localLowIndex + 1 + i] = inputSignal[yIndexLocal * stride + (lowCoefficientIndex - filterSideWidth + 1 + i) + indexOffset];
            //high
            indexOffset = mirrorIndexHigh[highCoefficientIndex + 1 + i];
            xOfff = (highCoefficientIndex - filterSideWidth + highCoefficientOffsetX + indexOffset + 1 + i);
            highVals[localHighIndex + 1 + i] =  inputSignal[yIndexLocal * stride + xOfff];

        }
    }
    __syncthreads();
#endif


    int64 offsetVals = OFFSETVAL;
    int64 i = offsetVals;
    while(yLowStart > -1) {
        double filterVal = 0;
        double valLow = 0;
#if defined SHARED_MEMORY
        filterVal = sLowfilter[yLowStart];
        valLow = lowVals[localLowIndex + i];
#else
        filterVal = lowFilter[yLowStart];
        if(lowCoefficientIndex + i < 4 || lowCoefficientIndex + i >= blockWidth / 2 + 4) {
            int64 indexOffset = mirrorIndexLow[lowCoefficientIndex + i];
            valLow = inputSignal[yIndexLocal * stride + (lowCoefficientIndex - filterSideWidth + i) + indexOffset];
        } else {
            valLow = inputSignal[yIndexLocal * stride + (lowCoefficientIndex - filterSideWidth + i)];
        }
#endif
        sum += filterVal * valLow;
        yLowStart -= 2;
        i++;
    }

    //high

    i = offsetVals;
    while(yHighStart > -1) {
        double filterVal = 0;
        double valHigh = 0;
#if defined SHARED_MEMORY
        filterVal = sHighfilter[yHighStart];
        valHigh = highVals[localHighIndex + i];
#else
        filterVal = highFilter[yHighStart];
        int64 xComponent = 0;
        if(highCoefficientIndex + i < 4 || highCoefficientIndex + i >= blockWidth / 2 + 4) {
            int64 indexOffset = mirrorIndexHigh[highCoefficientIndex + i];
            xComponent = (highCoefficientIndex - filterSideWidth + highCoefficientOffsetX + i + indexOffset);
        } else {
            xComponent = (highCoefficientIndex - filterSideWidth + highCoefficientOffsetX + i);
        }
        valHigh = inputSignal[yIndexLocal * stride + xComponent];
#endif
        sum += filterVal * valHigh;
        i++;
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
    int64 lowCoefficientIndex = (yIndexLocal + 1) / 2;

    int64 offsetVals = OFFSETVAL;
    int64 i = offsetVals;
    while(yLowStart > -1) {
        double filterVal = 0;
#if defined SHARED_MEMORY
        filterVal = sLowfilter[yLowStart];
#else
        filterVal = lowFilter[yLowStart];
#endif
        double valLow = 0;
        if(lowCoefficientIndex + i < 4 || lowCoefficientIndex + i >= blockHeight / 2 + 4) {
            int64 indexOffset = mirrorIndexLow[lowCoefficientIndex + i];
            valLow = inputSignal[(lowCoefficientIndex - filterSideWidth + i + indexOffset) * stride
                                     + xIndexLocal ];
        } else {
            valLow = inputSignal[(lowCoefficientIndex - filterSideWidth + i) * stride + xIndexLocal];
        }
        sum += filterVal * valLow;
        yLowStart -= 2;
        i++;
    }

    //high
    int64 highCoefficientIndex = yIndexLocal / 2;
    
    i = OFFSETVAL;
    double valHigh = 0;
    while(yHighStart > -1) {
        double filterVal = 0;
#if defined SHARED_MEMORY
        filterVal = sHighfilter[yHighStart];
#else
        filterVal = highFilter[yHighStart];
#endif
        if(highCoefficientIndex + i < 4 || highCoefficientIndex + i >= blockHeight / 2 + 4) {
            int64 indexOffset = mirrorIndexHigh[highCoefficientIndex + i];
            valHigh = inputSignal[(highCoefficientIndex + highCoefficientOffsetY - filterSideWidth + i + indexOffset) * stride
                                      + xIndexLocal ];
        } else {
            valHigh = inputSignal[(highCoefficientIndex + highCoefficientOffsetY - filterSideWidth + i) * stride
                                      + xIndexLocal ];
        }
        sum += filterVal * valHigh;
        yHighStart -= 2;
        i++;
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
            char * deviceIndexHigh) {

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
                    deviceOutputCoefficients, deviceIndexLow, deviceIndexHigh);
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
