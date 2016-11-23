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

__device__ int64 translateToRealIndex(struct ImageMeta inputSize, int64 index) {
    int64 realStride = inputSize.imageWidth;

    int64 inputStride = inputSize.xEnd - inputSize.xStart;
    int64 y = index / inputStride;
    int64 x = index % inputStride;
    return y * realStride + inputSize.xStart + x;
}

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

    if(xIndex <= sideWidth) {

        extendedSignal[index] = 0;

    } else if(xIndex >= sideWidth && xIndex <= inputStride - sideWidth) {

        int64 inputIndex = (extendedInputSize.yStart + yIndex) * realStride + (xIndex - sideWidth) + extendedInputSize.xStart;
        extendedSignal[index] = inputSignal[inputIndex];

    }  else {
        //extendedSignal[index] = SIGNAL_PAD_VALUE;
        extendedSignal[index] = 0;
    }
}

int64 calculateExtendedSignalLength(int64 width, int64 height, int64 filterSize) {
    int64 filterSideSize = filterSize / 2;
    int64 extendedWidth = width + filterSideSize * 2;
    return extendedWidth * height; 
}

void dwt2D_Horizontal(MyVector & L, int levelsToCompress,
                      double * deviceInputSignal, 
                      struct ImageMeta &  inputImageMeta,
                      double * deviceLowFilter,
                      double * deviceHighFilter,
                      int64 filterLength,
                      struct ImageMeta & outputImageMeta,
                      double * deviceOutputCoefficients) {
    //allocate output max memory size
    int blockWidth = inputImageMeta.imageWidth; 
    int blockHeight = inputImageMeta.imageHeight; 
    int64 extendedImageSize = calculateExtendedSignalLength(blockWidth, blockHeight, filterLength);

    double * deviceTmpMemory = initTmpCoefficientMemory(extendedImageSize);
        
    //calculate extended Image Meta
    struct ImageMeta extendedImageMeta = inputImageMeta;
    int64 extendedWidth = (filterLength / 2) * 2;
    extendedImageMeta.xEnd += extendedWidth;
    extendedImageMeta.imageWidth += extendedWidth;

    //extend the image
    int threads;
    dim3 blocks;
    calculateBlockSize(extendedImageSize, threads, blocks);
    extend2D_Horizontal<<<blocks, threads>>>(inputImageMeta, extendedImageMeta, deviceInputSignal, 
                                             deviceTmpMemory, filterLength);
    std::cerr<<threads<<std::endl;
    debugTmpMemory(deviceTmpMemory, extendedImageSize, extendedImageMeta.imageWidth);
    //initlize output memory
    //int extendedMemorySize = blockWidth * blockHeight;
    //double * deviceOutputMemory = initTmpCoefficientMemory(extendedMemorySize);
    cudaFree(deviceTmpMemory);
}
