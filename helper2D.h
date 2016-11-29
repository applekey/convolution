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
__global__ void extend2D_Horizontal(struct ImageMeta origionalImageSize,
                                    struct ImageMeta extendedInputSize, double * inputSignal,
                                    double * extendedSignal, int64 filterSize) {
    int64 index = calculateIndex();
    int64 totalBlockSize = extendedInputSize.imageWidth * extendedInputSize.imageHeight;

    if(index >= totalBlockSize) {
        return;
    }

    int64 realStride = origionalImageSize.imageWidth;
    int64 sideWidth = filterSize / 2;

    int64 inputStride = extendedInputSize.xEnd - extendedInputSize.xStart;
    int64 yIndex = index / inputStride;
    int64 xIndex = index % inputStride;

    if(xIndex < sideWidth) {

        int64 inputIndex = (extendedInputSize.yStart + yIndex) * realStride + 
                            extendedInputSize.xStart;
        extendedSignal[index] = inputSignal[inputIndex];

    } else if(xIndex >= sideWidth && xIndex < inputStride - sideWidth) {

        int64 inputIndex = (extendedInputSize.yStart + yIndex) * realStride + 
                           (xIndex - sideWidth) + extendedInputSize.xStart;
        extendedSignal[index] = inputSignal[inputIndex];

    }  else {
        //extendedSignal[index] = SIGNAL_PAD_VALUE;
        int64 inputIndex = (extendedInputSize.yStart + yIndex) * realStride + 
                            extendedInputSize.xStart + inputStride -1;
        extendedSignal[index] = inputSignal[inputIndex];
    }
}

__global__ void convolve2D_Horizontal(double * inputSignal, int signalLength,
                                       double * filter, int filterLength,
                                       double * output, struct ImageMeta outputImageMeta) {
    int64 index = calculateIndex();
    int64 inputIndex = index * 2 + (filterLength / 2);

    double sum = 0.0;

    for(int64 i = 0; i < filterLength; i++) {
        sum += filter[i] * inputSignal[inputIndex - (filterLength / 2) + i];
    }

    int64 outputIndex = translateToRealIndex(outputImageMeta, index, true);
    output[outputIndex] = sum;
}

/*---------------------------------VERT---------------------------------------*/ 
__global__ void extend2D_Vertical(struct ImageMeta origionalImageSize,
                                    struct ImageMeta extendedInputSize, double * inputSignal,
                                    double * extendedSignal, int64 filterSize) {
    int64 index = calculateIndex();
    int64 totalBlockSize = extendedInputSize.imageWidth * extendedInputSize.imageHeight;

    if(index >= totalBlockSize) {
        return;
    }
    int64 realStride = origionalImageSize.imageWidth;

    int64 sideWidth = filterSize / 2;

    int64 inputStride = extendedInputSize.xEnd - extendedInputSize.xStart;
    int64 yIndex = index / inputStride;
    int64 xIndex = index % inputStride;
    int64 yInputStride = extendedInputSize.yEnd - extendedInputSize.yStart;

    int64 rotatedIndex = xIndex * yInputStride + yIndex; 
    if(yIndex < sideWidth) {

        int64 inputIndex = (extendedInputSize.yStart) * realStride + 
                           xIndex + extendedInputSize.xStart;

        extendedSignal[rotatedIndex] = inputSignal[inputIndex];
        //extendedSignal[rotatedIndex] = 0.3;

    } else if(yIndex >= sideWidth && yIndex < yInputStride - sideWidth) {

        int64 inputIndex = (extendedInputSize.yStart + yIndex - sideWidth ) * realStride + 
                           xIndex + extendedInputSize.xStart;
        extendedSignal[rotatedIndex] = inputSignal[inputIndex];

    }  else {
        int64 inputIndex = (extendedInputSize.yStart + yInputStride - 1 - sideWidth * 2) * realStride + 
                           xIndex + extendedInputSize.xStart;
        extendedSignal[rotatedIndex] = inputSignal[inputIndex];
        //extendedSignal[rotatedIndex] = 1.3;
    }
}

__global__ void convolve2D_Vertical(double * inputSignal, int signalLength,
                                       double * filter, int filterLength,
                                       double * output, struct ImageMeta outputImageMeta) {
    int64 index = calculateIndex();
    int64 inputIndex = index * 2 + (filterLength / 2);

    double sum = 0.0;

    for(int64 i = 0; i < filterLength; i++) {
        sum += filter[i] * inputSignal[inputIndex - (filterLength / 2) + i];
    }

    int64 outputIndex = translateToRealIndex(outputImageMeta, index, false);
    output[outputIndex] = sum;
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
                      double * deviceOutputCoefficients) {

    struct ImageMeta currentImageMeta = inputImageMeta;
    double * currentInputSignal = deviceInputSignal;

    //allocate output max memory size
    int blockWidth = inputImageMeta.imageWidth; 
    int blockHeight = inputImageMeta.imageHeight; 
    int64 extendedImageSize = calculateExtendedSignalLength(blockWidth, blockHeight, filterLength, true);

    double * deviceTmpMemory = initTmpCoefficientMemory(extendedImageSize);

    bool isHorizontal = true;
        
    for(int i = 0; i < levelsToCompress; i++) {

        //calculate extended Image Meta horizontal / vert
        struct ImageMeta extendedImageMeta = currentImageMeta;
        int64 extendedLength = (filterLength / 2) * 2;

        if(isHorizontal) {
            extendedImageMeta.xEnd += extendedLength;
            extendedImageMeta.imageWidth += extendedLength;
        } else {
            extendedImageMeta.yEnd += extendedLength;
            extendedImageMeta.imageHeight += extendedLength;
        }

        //extend the image horizontal / vert 
        int threads;
        dim3 blocks;
        calculateBlockSize(extendedImageSize, threads, blocks);
        if(isHorizontal) {
            extend2D_Horizontal<<<blocks, threads>>>(inputImageMeta, extendedImageMeta, currentInputSignal, 
                                                     deviceTmpMemory, filterLength);
            int64 extendedWidth =  extendedImageMeta.xEnd - extendedImageMeta.xStart;
            //debugTmpMemory(deviceTmpMemory, extendedImageSize, extendedWidth);
        } else {
            extend2D_Vertical<<<blocks, threads>>>(inputImageMeta, extendedImageMeta, currentInputSignal, 
                                                     deviceTmpMemory, filterLength);
            int64 extendedHeight =  extendedImageMeta.yEnd - extendedImageMeta.yStart;
            //debugTmpMemory(deviceTmpMemory, extendedImageSize, extendedHeight);
        }

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
            convolve2D_Horizontal<<<blocks, threads>>> (deviceTmpMemory, convolveImagSize, 
                                                        deviceLowFilter, filterLength,
                                                        deviceOutputCoefficients, imageMetaLow);

            //high filter
            convolve2D_Horizontal<<<blocks, threads>>> (deviceTmpMemory, convolveImagSize, 
                                                        deviceHighFilter, filterLength,
                                                        deviceOutputCoefficients, imageMetaHigh);
        } else {
            //low filter
            convolve2D_Vertical<<<blocks, threads>>> (deviceTmpMemory, convolveImagSize, 
                                                        deviceLowFilter, filterLength,
                                                        deviceOutputCoefficients, imageMetaLow);

            //high filter
            convolve2D_Vertical<<<blocks, threads>>> (deviceTmpMemory, convolveImagSize, 
                                                        deviceHighFilter, filterLength,
                                                        deviceOutputCoefficients, imageMetaHigh);
        } 

        if(!isHorizontal) {
            //inputImageMeta, width and height divide by 2
            currentImageMeta.xEnd /= 2; 
            currentImageMeta.yEnd /= 2; 
            blockWidth = currentImageMeta.xEnd - currentImageMeta.xStart;
            blockHeight = currentImageMeta.yEnd - currentImageMeta.yStart;
            extendedImageSize = calculateExtendedSignalLength(blockWidth, blockHeight, filterLength, isHorizontal);
        }
        currentInputSignal = deviceOutputCoefficients;
        isHorizontal = !isHorizontal;
    }
    cudaFree(deviceTmpMemory);
}   

//void dwt2D_Horizontal(MyVector & L, int levelsToCompress,
                      //double * deviceInputSignal, 
                      //struct ImageMeta & inputImageMeta,
                      //double * deviceLowFilter,
                      //double * deviceHighFilter,
                      //int64 filterLength,
                      //struct ImageMeta & outputImageMeta,
                      //double * deviceOutputCoefficients) {

//}
