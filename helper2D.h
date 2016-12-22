#include "helper.h"
#define MAX_SIDE 256
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
__global__ void convolve2D_Horizontal(double * inputSignal, int signalLength,
                                      double * lowFilter, double * highFilter, int filterLength,
                                      double * output, struct ImageMeta inputImageMeta, int64 offset,
                                      int * mirrorIndex) {
    int64 index = calculateIndex();

    if (index >= signalLength) {
        return;
    }
    int64 imageWidth = inputImageMeta.imageWidth;
    int64 stride = inputImageMeta.xEnd;

    int64 yIndex = (index) / stride;
    int64 xIndex = (index) % stride;

    int64 filterSideWidth = filterLength / 2;

#if defined SHARED_MEMORY
    __shared__ double sLowfilter[9]; //max per
    __shared__ double sHighfilter[9]; //max per
    if(int64(threadIdx.x) < int64(9)) {
        sLowfilter[threadIdx.x] = lowFilter[threadIdx.x];
        sHighfilter[threadIdx.x] = highFilter[threadIdx.x];
    }

    int64 indexOffset = mirrorIndex[xIndex];

    __shared__ double s[1024 + 8]; //max per
    if(stride >= 1024) {
        s[threadIdx.x] = inputSignal[yIndex * imageWidth + xIndex - filterSideWidth + indexOffset];
        if(threadIdx.x == 1023) {
            int64 startT = 1024;
            //there might be a bug here
            for(int i = 0; i < 8; i++) {
                indexOffset = mirrorIndex[xIndex  + 1 + i];
                s[startT + i] = inputSignal[yIndex * imageWidth + xIndex - filterSideWidth + indexOffset + i + 1];
            }
        }
    }

    __syncthreads();
#endif

    double vals[9];

    for (int i = 0; i < 9 ; i++) {
#if defined SHARED_MEMORY
        if(stride >= 1024) {
            vals[i] = s[threadIdx.x + i];
        } else {
            int64 indexOffset = mirrorIndex[xIndex  + i];
            vals[i] = inputSignal[yIndex * imageWidth + xIndex - filterSideWidth + i + indexOffset];
        }
#else
        int64 indexOffset = mirrorIndex[xIndex  + i];
        vals[i] = inputSignal[yIndex * imageWidth + xIndex - filterSideWidth + i + indexOffset];
#endif
    }

    double * filter;
    double sum = 0.0;
    if(xIndex % 2 == 0) {
#if defined SHARED_MEMORY
        filter = sLowfilter;
#else
        filter = lowFilter;
#endif
        offset = 0;
        //-4
        sum += vals[0] * filter[0];
        //4
        sum += vals[8] * filter[8];
    } else {
#if defined SHARED_MEMORY
        filter = sHighfilter;
#else
        filter = highFilter;
#endif
    }

    //-3
    sum += vals[1] * filter[1];
    //-2
    sum += vals[2] * filter[2];
    //-1
    sum += vals[3] * filter[3];
    //0
    sum += vals[4] * filter[4];
    //1
    sum += vals[5] * filter[5];
    //2
    sum += vals[6] * filter[6];
    //3
    sum += vals[7] * filter[7];

    output[yIndex * imageWidth + xIndex / 2 + offset] = sum;
}
__global__ void convolve2D_Vertical(double * inputSignal, int signalLength,
                                    double * lowFilter, double * highFilter,
                                    int filterLength,
                                    double * output, struct ImageMeta inputImageMeta,
                                    int64 offset, int64 maxThreadWidth,
                                    int * mirrorIndex) {
    int64 origIndex = calculateIndex();
    int64 index = origIndex;

    if (index >= signalLength) {
        return;
    }

    int64 imageWidth = inputImageMeta.imageWidth;
    int64 height = inputImageMeta.yEnd;

    //order is using blocks of maxThreadWidth

    int64 yRoll =  index / maxThreadWidth;
    int64 yIndex = yRoll % height;
    int64 yRollX =  yRoll / height;
    int64 xIndex = (index % maxThreadWidth) + maxThreadWidth * yRollX;

    //old linear method
    //int64 yIndex = index / stride * 2 + highOffset;
    //int64 xIndex = index % stride;

    int64 filterSideWidth = filterLength / 2;

#if defined SHARED_MEMORY
    __shared__ double sLowfilter[9]; //max per
    __shared__ double sHighfilter[9]; //max per
    if(int64(threadIdx.x) < int64(9)) {
        sLowfilter[threadIdx.x] = lowFilter[threadIdx.x];
        sHighfilter[threadIdx.x] = highFilter[threadIdx.x];
    }

    __shared__ int sIndexOffset[9 + (1024 / MAX_SIDE - 1)];

    if(threadIdx.x < 9 + 1024 / MAX_SIDE) {
        sIndexOffset[threadIdx.x] = mirrorIndex[yIndex + threadIdx.x];
    }

    __syncthreads();

    __shared__ double s[MAX_SIDE * (9 + (1024 / MAX_SIDE - 1))]; //max per
    int iOffset = threadIdx.x / MAX_SIDE;
    int numPerT =  (9 + (1024 / MAX_SIDE - 1) / (1024 / MAX_SIDE));

    for(int i = iOffset * numPerT; i < (iOffset + 1) * numPerT; i++) {
        if( i >= (9 + (1024 / MAX_SIDE - 1))) {
            break;
        }
        int64 indexOffset = sIndexOffset[i];
        s[threadIdx.x % MAX_SIDE + i * MAX_SIDE] = inputSignal[((yIndex - iOffset) - filterSideWidth + i + indexOffset) * imageWidth
                + xIndex ];
    }

    __syncthreads();
#endif

    double vals[9];

    for (int i = 0; i < 9; i++) {
#if defined SHARED_MEMORY
        vals[i] = s[threadIdx.x + i * MAX_SIDE];
#else
        int64 indexOffset = mirrorIndex[yIndex + i];
        vals[i] = inputSignal[(yIndex - filterSideWidth + i + indexOffset) * imageWidth + xIndex ];
#endif
    }

    double * filter;
    double sum = 0.0;

    if(yIndex % 2 == 0) {
#if defined SHARED_MEMORY
        filter = sLowfilter;
#else
        filter = lowFilter;
#endif
        offset = 0;
        //-4
        sum += vals[0] * filter[0];
        ////4
        sum += vals[8] * filter[8];
    } else {
#if defined SHARED_MEMORY
        filter = sHighfilter;
#else
        filter = highFilter;
#endif
    }

    //-3
    sum += vals[1] * filter[1];
    //-2
    sum += vals[2] * filter[2];
    //-1
    sum += vals[3] * filter[3];
    //0
    sum += vals[4] * filter[4];
    //1
    sum += vals[5] * filter[5];
    //2
    sum += vals[6] * filter[6];
    ////3
    sum += vals[7] * filter[7];

    output[(yIndex / 2 + offset)* imageWidth + xIndex] = sum;
}

int64 calculateExtendedSignalLength(int64 width, int64 height, int64 filterSize,
                                    bool isHorizontal) {
    int64 filterSideSize = filterSize / 2;
    return (isHorizontal) ? width * (height + filterSideSize * 2) :
           height * (width + filterSideSize * 2) ;
}

struct ImageMeta dwt2D(int levelsToCompress,
                       double * deviceInputSignal,
                       struct ImageMeta inputImageMeta,
                       double * deviceLowFilter,
                       double * deviceHighFilter,
                       int64 filterLength,
                       struct ImageMeta & outputImageMeta,
                       double * deviceOutputCoefficients,
                       double * deviceTmpMemory,
                       int * mirrorIndex, int * hostIndex) {

    struct ImageMeta currentImageMeta = inputImageMeta;
    double * currentInputSignal = deviceInputSignal;

    //allocate output max memory size
    int blockWidth = inputImageMeta.imageWidth;
    int blockHeight = inputImageMeta.imageHeight;

    bool isHorizontal = true;

    for (int level = 0; level < levelsToCompress * 2; level++) {

        int threads;
        dim3 blocks;
        //set up output image meta
        int64 convolveImagSize = convolveImagSize = blockWidth * blockHeight;

        //convolve the image
        calculateBlockSize(convolveImagSize, threads, blocks);
        //auto startLocal = std::chrono::system_clock::now();

        if (isHorizontal) {
            //low filter
            convolve2D_Horizontal <<< blocks, threads>>> (currentInputSignal, convolveImagSize,
                    deviceLowFilter, deviceHighFilter, filterLength,
                    deviceTmpMemory, currentImageMeta, blockWidth/2, mirrorIndex);

            currentInputSignal = deviceTmpMemory;

        } else {
            //low filter
            int64 maxThreadWidth = blockWidth;

            if(maxThreadWidth > MAX_SIDE) {
                maxThreadWidth = MAX_SIDE;
            }

            convolve2D_Vertical <<< blocks, threads>>> (currentInputSignal, convolveImagSize,
                    deviceLowFilter, deviceHighFilter, filterLength,
                    deviceOutputCoefficients, currentImageMeta, blockHeight / 2, maxThreadWidth,
                    mirrorIndex);

        }

        //cudaDeviceSynchronize();
        //auto endLocal = std::chrono::system_clock::now();
        //std::chrono::duration<double> diff = endLocal - startLocal;
        //std::cerr<<"DWT level:"<<level/2<<" size: "<<currentImageMeta.xEnd<<", calc: "<<diff.count()<<std::endl;
        if (!isHorizontal) {
            //inputImageMeta, width and height divide by 2
            currentImageMeta.xEnd /= 2;
            currentImageMeta.yEnd /= 2;
            blockWidth = currentImageMeta.xEnd;
            blockHeight = currentImageMeta.yEnd;
            currentInputSignal = deviceOutputCoefficients;
            // move index coefficients
            hostIndex[currentImageMeta.xEnd] = hostIndex[0];
            hostIndex[currentImageMeta.xEnd + 1] = hostIndex[1];
            hostIndex[currentImageMeta.xEnd + 2] = hostIndex[2];
            hostIndex[currentImageMeta.xEnd + 3] = hostIndex[3];
            hostIndex += currentImageMeta.xEnd;
            mirrorIndex += currentImageMeta.xEnd;
            cudaMemcpy(mirrorIndex, hostIndex, sizeof(int) * 4, cudaMemcpyHostToDevice);
        }
        isHorizontal = !isHorizontal;

    }
    return currentImageMeta;
}
/*---------------------------INVERSE-------------------------*/
__global__ void inverseConvolveVertical(double * inputSignal, int64 filterLength,
                                        int64 totalSignalLength,
                                        double * lowFilter, double * highFilter,
                                        struct ImageMeta inputImageMeta,
                                        double * reconstructedSignal,
                                        int64 maxThreadWidth) {
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
        sum += lowFilter[yLowStart] * valsLow[iC + offsetVals];
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
        sum += highFilter[yHighStart] * valsHigh[iC + offsetVals];
        yHighStart -= 2;
        iC ++;
    }

    int64 outputIndex = yIndexLocal * stride + xIndexLocal;
    reconstructedSignal[outputIndex] = sum;
}

__global__ void inverseConvolveHorizontal(double * inputSignal, int64 filterLength,
        int64 totalSignalLength,
        double * lowFilter, double * highFilter,
        struct ImageMeta inputImageMeta,
        double * reconstructedSignal) {
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

    //low
    double valsLow[9];
    int64 lowCoefficientIndex = (xIndexLocal + 1) / 2;

    int fillLeft = filterSideWidth - lowCoefficientIndex;
    int filledL = 0;

    for (int i = 0; i < fillLeft; i++) {
        int64 mirrorDistance = fillLeft - i - LOW_LEFT;
        valsLow[i] = inputSignal[yIndexLocal * stride + mirrorDistance ];
        filledL += 1;
    }

    int fillRight = lowCoefficientIndex - (highCoefficientOffsetX - filterSideWidth - 1);
    int filledR = 0;
    for (int i = 0; i < fillRight; i++) {
        int64 mirrorDistance = fillRight - i - LOW_RIGHT;
        valsLow[8 - i] = inputSignal[yIndexLocal * stride + (highCoefficientOffsetX - 1 - mirrorDistance) ];
        filledR += 1;
    }

    for (int i = filledL; i < 9 - filledR; i++) {
        valsLow[i] = inputSignal[yIndexLocal * stride + (lowCoefficientIndex - filterSideWidth + i) ];
    }
    int64 offsetVals = OFFSETVAL;
    int64 iC = 0;
    while(yLowStart > -1) {
        sum += lowFilter[yLowStart] * valsLow[iC + offsetVals];
        yLowStart -= 2;
        iC++;
    }

    //high
    double valsHigh[9];
    int64 highCoefficientIndex = xIndexLocal / 2;
    fillLeft = filterSideWidth - highCoefficientIndex;
    filledL = 0;

    for (int i = 0; i < fillLeft; i++) {
        int64 mirrorDistance = fillLeft - i - HIGH_LEFT;
        valsHigh[i] = inputSignal[yIndexLocal * stride + (highCoefficientOffsetX + mirrorDistance) ];
        filledL += 1;
    }

    fillRight = highCoefficientIndex - (highCoefficientOffsetX - filterSideWidth - 1);
    filledR = 0;
    for (int i = 0; i < fillRight; i++) {
        int64 mirrorDistance = fillRight - i - HIGH_RIGHT;
        valsHigh[8 - i] = inputSignal[yIndexLocal * stride + (2 * highCoefficientOffsetX - 1 - mirrorDistance) ];
        filledR += 1;
    }

    for (int i = filledL; i < 9 - filledR; i++) {
        valsHigh[i] = inputSignal[yIndexLocal * stride + (highCoefficientIndex - filterSideWidth + i + highCoefficientOffsetX) ];
    }

    iC = 0;
    while(yHighStart > -1) {
        sum += highFilter[yHighStart] * valsHigh[iC+ offsetVals];
        iC++;
        yHighStart -= 2;
    }

    int64 outputIndex = yIndexLocal * stride + xIndexLocal;
    reconstructedSignal[outputIndex] = sum;
}

void iDwt2D(int levelsToCompressUncompress,
            double * deviceInputSignal,
            struct ImageMeta & inputImageMeta,
            double * deviceILowFilter,
            double * deviceIHighFilter,
            int64 filterLength,
            double * deviceOutputCoefficients,
            double * deviceTmpMemory) {

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
                    deviceOutputCoefficients);
        } else {
            //expand current image size
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
                    maxThreadWidth);
        }
        //cudaDeviceSynchronize();

        auto endLocal = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = endLocal - startLocal;
        //std::cerr<<"DWT-I level: "<<level/2<<" size: "<<currentImageMeta.xEnd<<", calc: "<<diff.count()<<std::endl;

        isHorizontal = !isHorizontal;
    }
}
