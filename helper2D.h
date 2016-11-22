#include "helper.h"

struct ImageMeta {
    int64 imageWidth, imageHeight;
    int64 xStart, xEnd;
    int64 yStart, yEnd;
};

struct vec2 {
    vec2(int x, int y) {
        x = x;
        y = y;
    }
    int64 x, y;
};

struct vec2 get2DIndex() {
    int x = -1;
    int y = -1;
    return vec2(x, y); 
}

//extend the signal
__void__ int64 calculateIndex() {
}
__device__ void convolve2D() {
    struct vec2 index = get2DIndex();

    int64 inputIndex = index * 2 + (filterLength / 2);

    double sum = 0.0;

    for(int64 i = 0; i < filterLength; i++) {
        sum += filter[i] * inputSignal[inputIndex - (filterLength / 2) + i];
    }

    output[index + outputOffset] = sum;
}



int calcualteNewIndex() {
    if i > bounds modulo thing
    then nxt line i guess
    else
    index offset ++ 
}
calculateOutputIndexHorizontal() {
    offset = half of the image
   horizontal index / w2 
    verticle index the same
    
}

calculateOutputIndexVerticle() {
    

}

__device__ int64 calculateExtendedSignalLength(int64 width, int64 height, int filterSize) {
    int filterSideSize = filterSize / 2;
    int64 extendedWidth = width + filterSideSize * 2;
    int64 extendedHeight = height + filterSideSize * 2;
    return extendedWidth * extendedWidth; 
}
__device__ void extend2D(struct vec2 inputSize, double * output) {
    //figure out 
    //copy into a linear memory array
}

void dwt2D_Horizontal(MyVector & L, int levelsToCompress,
                      double * deviceInputSignal, 
                      struct ImageMeta &  inputImageMeta,
                      double * deviceLowFilter,
                      double * deviceHighFilter,
                      int64 filterLength,
                      struct ImageMeta & outputImageMeta,
                      double * deviceOutputCoefficients) {
    //allocate output memory
    int blockWidth = 
    int blockHeight =  
    int extendedMemorySize = calculateExtendedSignalLength();
    double * deviceTmpMemory = initTmpCoefficientMemory();
        
    //extend the signal 


    //initlize output memory
    int extendedMemorySize = widht * height;
    double * deviceTmpMemory = initTmpCoefficientMemory();
}
