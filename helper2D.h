#include "helper.h"


struct vec2 {
    vec2(int x, int y) {
        x = x;
        y = y;
    }
    int64 x, y;
};

//struct vec2 get2DIndex() {
    //int x = -1;
    //int y = -1;
    //return vec2(x, y); 
//}

////extend the signal
//__void__ int64 calculateIndex() {
//}
//__device__ void convolve2DHorizontal() {
    //struct vec2 index = get2DIndex();

    //int64 inputIndex = index * 2 + (filterLength / 2);

    //double sum = 0.0;

    //for(int64 i = 0; i < filterLength; i++) {
        //sum += filter[i] * inputSignal[inputIndex - (filterLength / 2) + i];
    //}

    //output[index + outputOffset] = sum;
//}



//int calcualteNewIndex() {
    //if i > bounds modulo thing
    //then nxt line i guess
    //else
    //index offset ++ 
//}

//calculateOutputIndexHorizontal() {
    //offset = half of the image
   //horizontal index / w2 
    //verticle index the same
    //calcualteNewIndex
//}

//calculateOutputIndexVerticle() {
    

//}

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

__global__ void extend2D_Horizontal(struct ImageMeta extendedInputSize, double * inputSignal,
                            double * extendedSignal, int64 filterSize) {
    int64 index = calculateIndex();
    int64 totalBlockSize = extendedInputSize.imageWidth * extendedInputSize.imageHeight;

    if(index >= totalBlockSize) {
        return;
    }

    int64 realStride = extendedInputSize.imageWidth;
    int64 sideWidth = filterSize / 2;

    int64 inputStride = extendedInputSize.xEnd - extendedInputSize.xStart;
    int64 yIndex = index / inputStride;
    int64 xIndex = index % inputStride;

    if(xIndex < sideWidth) {

        extendedSignal[index] = 0;

    } else if(index >= sideWidth && index < inputStride - sideWidth) {

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
    int64 extendedHeight = height + filterSideSize * 2;
    return extendedWidth * extendedHeight; 
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
    int64 extendedMemorySize = calculateExtendedSignalLength(blockWidth, blockHeight, filterLength);
    double * deviceTmpMemory = initTmpCoefficientMemory(extendedMemorySize);
        
    //calculate extended Image Meta
    struct ImageMeta extendedImageMeta = inputImageMeta;
    extendedImageMeta.xStart -= filterLength / 2;
    extendedImageMeta.xEnd += filterLength / 2;
    

    //initlize output memory
    //int extendedMemorySize = blockWidth * blockHeight;
    //double * deviceOutputMemory = initTmpCoefficientMemory(extendedMemorySize);
}
