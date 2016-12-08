#include "helper.h"


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

__device__ int64 translateToRealIndex(struct ImageMeta inputSize, int64 index,
                                      bool isHorizontal) {
    int64 realStride = inputSize.imageWidth;

    int64 inputStride = (isHorizontal) ? (inputSize.xEnd - inputSize.xStart): 
                                         (inputSize.yEnd - inputSize.yStart);
    
    int64 x,y;
    if(isHorizontal) {
        y = index / inputStride;
        x = index % inputStride;
    } else {
        x = index / inputStride;
        y = index % inputStride;
    }
    return (y + inputSize.yStart) * realStride + inputSize.xStart + x;
}

/*------------------------------------------------------HORIZONTAL------------------------------*/
__global__ void convolve2D_Horizontal(double * inputSignal, int signalLength,
                                       double * filter, int filterLength,
                                       double * output, struct ImageMeta outputImageMeta, int64 offset) {
    int64 index = calculateIndex();

    int64 stride = outputImageMeta.imageWidth;
    int64 height = outputImageMeta.imageHeight;

    int64 yIndex = index * 2/ stride;
    int64 xIndex = index * 2 % stride;

    int64 filterSideWidth = filterLength / 2;

    double vals[9];

    int fillLeft = filterSideWidth - xIndex;
    int filledL = 0;
    for(int i =0; i< fillLeft; i++) {
        vals[i] = 1.0;
        filledL += 1;
    } 

    int fillRight = xIndex - (stride - filterSideWidth - 1);
    int filledR = 0;
    for(int i =0; i< fillRight; i++) {
        vals[9 - i] = 1.0;
        filledR += 1;
    } 

    for(int i = filledL; i < 9 - filledR; i++) {
        vals[i] = inputSignal[yIndex * stride + xIndex - filterSideWidth + i ]; 
    }

    double sum = 0.0;
    //-4
    sum += vals[0] * filter[0];
    //-3
    sum += vals[1]* filter[1];
    //-2
    sum += vals[2]* filter[2];
    //-1
    sum += vals[3]* filter[3];
    //0
    sum += vals[4]* filter[4];
    //1
    sum += vals[5]* filter[5];
    //2
    sum += vals[6]* filter[6];
    //3
    sum += vals[7]* filter[7];
    //4
    sum += vals[8]* filter[8];

    output[yIndex * stride + xIndex/2 + offset] = sum;
}

/*---------------------------------VERT---------------------------------------*/ 
__global__ void convolve2D_Vertical(double * inputSignal, int signalLength,
                                       double * filter, int filterLength,
                                       double * output, struct ImageMeta outputImageMeta, int64 offset) {
    int64 origIndex = calculateIndex();
    int64 index = origIndex;

    
    int64 stride = outputImageMeta.imageWidth;
    int64 height = outputImageMeta.imageHeight;

    int64 yIndex = index / stride * 2;
    int64 xIndex = index % stride;

    int64 filterSideWidth = filterLength / 2;

    double vals[9];

    int fillLeft = filterSideWidth - yIndex;
    int filledL = 0;
    for(int i =0; i< fillLeft; i++) {
        vals[i] = 1.0;
        filledL += 1;
    } 

    int fillRight = yIndex - (height - filterSideWidth - 1);
    int filledR = 0;
    for(int i =0; i< fillRight; i++) {
        vals[9 - i] = 1.0;
        filledR += 1;
    } 

    for(int i = filledL; i < 9 - filledR; i++) {
        vals[i] = inputSignal[(yIndex - filterSideWidth + i)* stride + xIndex ]; 
    }

    double sum = 0.0;
    //-4
    sum += vals[0] * filter[0];
    //-3
    sum += vals[1]* filter[1];
    //-2
    sum += vals[2]* filter[2];
    //-1
    sum += vals[3]* filter[3];
    //0
    sum += vals[4]* filter[4];
    //1
    sum += vals[5]* filter[5];
    //2
    sum += vals[6]* filter[6];
    //3
    sum += vals[7]* filter[7];
    //4
    sum += vals[8]* filter[8];
    
    output[(yIndex/2 + offset)* stride + xIndex] = sum;
}

int64 calculateExtendedSignalLength(int64 width, int64 height, int64 filterSize,
                                     bool isHorizontal) {
    int64 filterSideSize = filterSize / 2;
    return (isHorizontal) ? width * (height + filterSideSize * 2) : 
                            height * (width + filterSideSize * 2) ; 
}

void dwt2D_Horizontal(MyVector & L, int levelsToCompress,
                      double * deviceInputSignal, 
                      struct ImageMeta & inputImageMeta,
                      double * deviceLowFilter,
                      double * deviceHighFilter,
                      int64 filterLength,
                      struct ImageMeta & outputImageMeta,
                      double * deviceOutputCoefficients,
                      double * deviceTmpMemory) {

    struct ImageMeta currentImageMeta = inputImageMeta;
    double * currentInputSignal = deviceInputSignal;
    
    //allocate output max memory size
    int blockWidth = inputImageMeta.imageWidth; 
    int blockHeight = inputImageMeta.imageHeight; 

    bool isHorizontal = true;
        
    for(int i = 0; i < levelsToCompress; i++) {

        int threads;
        dim3 blocks;
        //set up output image meta
        struct ImageMeta imageMetaHigh = outputImageMeta;
        struct ImageMeta imageMetaLow = outputImageMeta;
        int64 convolveImagSize = convolveImagSize = blockWidth * blockHeight / 2; 

        if(isHorizontal) {
            imageMetaHigh.yStart = imageMetaHigh.yStart + blockHeight / 2; 
            imageMetaLow.yEnd = imageMetaHigh.yStart + blockHeight / 2;
        } else {
            imageMetaHigh.xStart = imageMetaHigh.xStart + blockHeight / 2;
            imageMetaLow.xEnd = imageMetaHigh.xStart + blockHeight / 2;
        }

        //convolve the image

        calculateBlockSize(convolveImagSize, threads, blocks);

        if (isHorizontal) {
            //low filter
            //deviceTmpMemory = deviceOutputCoefficients;
            convolve2D_Horizontal<<<blocks, threads>>> (currentInputSignal, convolveImagSize, 
                                                        deviceLowFilter, filterLength,
                                                        deviceTmpMemory, imageMetaLow, 0);

            ////high filter
            convolve2D_Horizontal<<<blocks, threads>>> (currentInputSignal, convolveImagSize, 
                                                        deviceHighFilter, filterLength,
                                                        deviceTmpMemory, imageMetaHigh, blockWidth/2);

            currentInputSignal = deviceTmpMemory; 
        } else {
            //low filter
            convolve2D_Vertical<<<blocks, threads>>> (currentInputSignal, convolveImagSize, 
                                                        deviceLowFilter, filterLength,
                                                        deviceOutputCoefficients, imageMetaLow, 0);

            //high filter
            convolve2D_Vertical<<<blocks, threads>>> (currentInputSignal, convolveImagSize, 
                                                        deviceHighFilter, filterLength,
                                                        deviceOutputCoefficients, imageMetaHigh, blockHeight/2);
        } 

        if(!isHorizontal) {
            //inputImageMeta, width and height divide by 2
            currentImageMeta.xEnd /= 2; 
            currentImageMeta.yEnd /= 2; 
            blockWidth = currentImageMeta.xEnd - currentImageMeta.xStart;
            blockHeight = currentImageMeta.yEnd - currentImageMeta.yStart;
            //extendedImageSize = calculateExtendedSignalLength(blockWidth, blockHeight, filterLength, isHorizontal);
            currentInputSignal = deviceOutputCoefficients;
        }
        isHorizontal = !isHorizontal;
    }
    //cudaFree(deviceTmpMemory);
}   
