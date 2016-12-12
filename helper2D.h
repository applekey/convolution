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

  int64 inputStride = (isHorizontal) ? (inputSize.xEnd - inputSize.xStart) :
                      (inputSize.yEnd - inputSize.yStart);

  int64 x, y;
  if (isHorizontal) {
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
                                      double * output, struct ImageMeta inputImageMeta, int64 offset) {
  int64 index = calculateIndex();

  if (index >= signalLength) {
    return;
  }

  int64 stride = inputImageMeta.xEnd - inputImageMeta.xStart;

  int64 yIndex = index * 2 / stride;
  int64 xIndex = index * 2 % stride;

  int64 filterSideWidth = filterLength / 2;

  double vals[9];

  int fillLeft = filterSideWidth - xIndex;
  int filledL = 0;
  for (int i = 0; i < fillLeft; i++) {
    vals[i]= inputSignal[yIndex * stride];
    filledL += 1;
  }

  int fillRight = xIndex - (stride - filterSideWidth) + 1;
  int filledR = 0;

  for (int i = 0; i < fillRight; i++) {
    vals[8 - i]= inputSignal[yIndex * stride + inputImageMeta.xEnd - 1];
    filledR += 1;
  }

  for (int i = filledL; i < 9 - filledR; i++) {
    vals[i] = inputSignal[yIndex * stride + xIndex - filterSideWidth + i ];
  }

  double sum = 0.0;
  //-4
  sum += vals[0] * filter[0];
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
  //4
  sum += vals[8] * filter[8];

  output[yIndex * stride + xIndex / 2 + offset] = sum;
}

/*---------------------------------VERT---------------------------------------*/
__global__ void convolve2D_Vertical(double * inputSignal, int signalLength,
                                    double * filter, int filterLength,
                                    double * output, struct ImageMeta inputImageMeta, int64 offset) {
  int64 origIndex = calculateIndex();
  int64 index = origIndex;

  if (index >= signalLength) {
    return;
  }

  int64 stride = inputImageMeta.xEnd - inputImageMeta.xStart;
  int64 height = inputImageMeta.yEnd - inputImageMeta.yStart;

  int64 yIndex = index / stride * 2;
  int64 xIndex = index % stride;

  int64 filterSideWidth = filterLength / 2;

  double vals[9];

  int fillLeft = filterSideWidth - yIndex;
  int filledL = 0;
  for (int i = 0; i < fillLeft; i++) {
    vals[i] = inputSignal[xIndex ];
    filledL += 1;
  }

  int fillRight = yIndex - (height - filterSideWidth) + 1;
  int filledR = 0;
  for (int i = 0; i < fillRight; i++) {
    vals[8 - i] = inputSignal[(height - 1) * stride + xIndex ];
    filledR += 1;
  }

  for (int i = filledL; i < 9 - filledR; i++) {
    vals[i] = inputSignal[(yIndex - filterSideWidth + i) * stride + xIndex ];
  }

  double sum = 0.0;
  //-4
  sum += vals[0] * filter[0];
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
  ////4
  sum += vals[8] * filter[8];

  output[(yIndex / 2 + offset)* stride + xIndex] = sum;
}

int64 calculateExtendedSignalLength(int64 width, int64 height, int64 filterSize,
                                    bool isHorizontal) {
  int64 filterSideSize = filterSize / 2;
  return (isHorizontal) ? width * (height + filterSideSize * 2) :
         height * (width + filterSideSize * 2) ;
}

struct ImageMeta dwt2D(MyVector & L, int levelsToCompress,
           double * deviceInputSignal,
           struct ImageMeta inputImageMeta,
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

  for (int i = 0; i < levelsToCompress; i++) {

    int threads;
    dim3 blocks;
    //set up output image meta
    struct ImageMeta imageMetaHigh = outputImageMeta;
    struct ImageMeta imageMetaLow = outputImageMeta;
    int64 convolveImagSize = convolveImagSize = blockWidth * blockHeight / 2;

    //if (isHorizontal) {
      //imageMetaHigh.yStart = imageMetaHigh.yStart + blockHeight / 2;
      //imageMetaLow.yEnd = imageMetaHigh.yStart + blockHeight / 2;
    //} else {
      //imageMetaHigh.xStart = imageMetaHigh.xStart + blockHeight / 2;
      //imageMetaLow.xEnd = imageMetaHigh.xStart + blockHeight / 2;
    //}

    //convolve the image
    calculateBlockSize(convolveImagSize, threads, blocks);

    if (isHorizontal) {
      //low filter
      convolve2D_Horizontal <<< blocks, threads>>> (currentInputSignal, convolveImagSize,
          deviceLowFilter, filterLength,
          deviceTmpMemory, imageMetaLow, 0);

      ////high filter
      convolve2D_Horizontal <<< blocks, threads>>> (currentInputSignal, convolveImagSize,
          deviceHighFilter, filterLength,
          deviceTmpMemory, imageMetaHigh, blockWidth / 2);

      currentInputSignal = deviceTmpMemory;
    } else {
      //low filter
      convolve2D_Vertical <<< blocks, threads>>> (currentInputSignal, convolveImagSize,
          deviceLowFilter, filterLength,
          deviceOutputCoefficients, imageMetaLow, 0);

      //high filter
      convolve2D_Vertical <<< blocks, threads>>> (currentInputSignal, convolveImagSize,
          deviceHighFilter, filterLength,
          deviceOutputCoefficients, imageMetaHigh, blockHeight / 2);
    }

    if (!isHorizontal) {
      //inputImageMeta, width and height divide by 2
      currentImageMeta.xEnd /= 2;
      currentImageMeta.yEnd /= 2;
      blockWidth = currentImageMeta.xEnd - currentImageMeta.xStart;
      blockHeight = currentImageMeta.yEnd - currentImageMeta.yStart;
      currentInputSignal = deviceOutputCoefficients;
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
                                        double * reconstructedSignal) {
  int64 index = calculateIndex();

  if (index >= totalSignalLength) {
    return;
  }

  int64 stride = inputImageMeta.imageWidth;
  int64 height = inputImageMeta.imageHeight;

  int64 blockWidth = stride;//(inputImageMeta.xEnd - inputImageMeta.xStart);//???ToDo fix here
  int64 yIndexLocal = index / blockWidth;
  int64 xIndexLocal = index % blockWidth;

  int64 filterSideWidth = filterLength / 2;

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
    valsLow[i] = inputSignal[(inputImageMeta.yStart) * stride + (inputImageMeta.xStart + xIndexLocal) ];
    //valsLow[i] = 1.0;
    filledL += 1;
  }

  int fillRight = lowCoefficientIndex - (highCoefficientOffsetY - filterSideWidth - 1);
  int filledR = 0;
  for (int i = 0; i < fillRight; i++) {
    valsLow[8 - i] = inputSignal[(inputImageMeta.yStart + highCoefficientOffsetY - 1) * stride
                                 + (inputImageMeta.xStart + xIndexLocal) ];
    //valsLow[9 - i] = 1.0;
    filledR += 1;
  }

  for (int i = filledL; i < 9 - filledR; i++) {
    valsLow[i] = inputSignal[(inputImageMeta.yStart + lowCoefficientIndex - filterSideWidth + i) * stride
                             + (inputImageMeta.xStart + xIndexLocal) ];
  }

  //lowIndex = filterLength - 1;
  //highIndex = filterLength - 2;
  //low = 8
  //high = 7
  //sum += lowFilter[0] * valsLow[0];
  sum += lowFilter[1] * valsLow[3];
  //sum += lowFilter[2] * valsLow[2];
  sum += lowFilter[3] * valsLow[2];
  //sum += lowFilter[4] * valsLow[4];
  sum += lowFilter[5] * valsLow[1];
  //sum += lowFilter[6] * valsLow[6];
  sum += lowFilter[7] * valsLow[0];
  //sum += 0 * valsLow[8];


  //high
  double valsHigh[9];
  int64 highCoefficientIndex = yIndexLocal / 2;
  fillLeft = filterSideWidth - highCoefficientIndex;
  filledL = 0;

  for (int i = 0; i < fillLeft; i++) {
    valsHigh[i] = inputSignal[(inputImageMeta.yStart + highCoefficientOffsetY) * stride
                              + (inputImageMeta.xStart + xIndexLocal) ];
    //valsHigh[i] = 1.0;
    filledL += 1;
  }

  fillRight = highCoefficientIndex - (highCoefficientOffsetY - filterSideWidth - 1);
  filledR = 0;
  for (int i = 0; i < fillRight; i++) {
    valsHigh[8 - i] = inputSignal[(inputImageMeta.yStart + 2 * highCoefficientOffsetY - 1) * stride
                                  + (inputImageMeta.xStart + xIndexLocal) ];
    filledR += 1;
  }

  for (int i = filledL; i < 9 - filledR; i++) {
    valsHigh[i] = inputSignal[(inputImageMeta.yStart + highCoefficientIndex + highCoefficientOffsetY - filterSideWidth + i) * stride
                              + (inputImageMeta.xStart + xIndexLocal) ];
  }

  sum += highFilter[0] * valsHigh[3];
  //sum += highFilter[1] * valsHigh[1];
  sum += highFilter[2] * valsHigh[2];
  //sum += highFilter[3] * valsHigh[3];
  sum += highFilter[4] * valsHigh[1];
  //sum += highFilter[5] * valsHigh[5];
  sum += highFilter[6] * valsHigh[0];
  //sum += highFilter[7] * valsHigh[7];
  //sum += highFilter[8] * valsHigh[8];

  int64 outputIndex = (yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart + xIndexLocal);
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
  int64 height = inputImageMeta.imageHeight;

  int64 blockWidth = (inputImageMeta.xEnd - inputImageMeta.xStart) * 2;
  int64 yIndexLocal = index / blockWidth;
  int64 xIndexLocal = index % blockWidth;

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
    valsLow[i] = inputSignal[(yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart) ];
    filledL += 1;
  }

  int fillRight = lowCoefficientIndex - (highCoefficientOffsetX - filterSideWidth - 1);
  int filledR = 0;
  for (int i = 0; i < fillRight; i++) {
    valsLow[8 - i] = inputSignal[(yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart + highCoefficientOffsetX - 1 ) ];
    filledR += 1;
  }

  for (int i = filledL; i < 9 - filledR; i++) {
    valsLow[i] = inputSignal[(yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart + lowCoefficientIndex - filterSideWidth + i) ];
  }

  //lowIndex = filterLength - 1;
  //highIndex = filterLength - 2;
  //low = 8
  //high = 7
  //sum += lowFilter[0] * valsLow[0];
  sum += lowFilter[1] * valsLow[3];
  //sum += lowFilter[2] * valsLow[2];
  sum += lowFilter[3] * valsLow[2];
  //sum += lowFilter[4] * valsLow[4];
  sum += lowFilter[5] * valsLow[1];
  //sum += lowFilter[6] * valsLow[6];
  sum += lowFilter[7] * valsLow[0];
  //sum += 0 * valsLow[8];


  //high
  double valsHigh[9];
  int64 highCoefficientIndex = xIndexLocal / 2;
  fillLeft = filterSideWidth - highCoefficientIndex;
  filledL = 0;

  for (int i = 0; i < fillLeft; i++) {
    valsHigh[i] = inputSignal[(yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart + highCoefficientOffsetX ) ];
    filledL += 1;
  }

  fillRight = highCoefficientIndex - (highCoefficientOffsetX - filterSideWidth - 1);
  filledR = 0;
  for (int i = 0; i < fillRight; i++) {
    valsHigh[8 - i] = inputSignal[(yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart + 2 * highCoefficientOffsetX - 1) ];
    filledR += 1;
  }

  for (int i = filledL; i < 9 - filledR; i++) {
    valsHigh[i] = inputSignal[(yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart + highCoefficientIndex - filterSideWidth + i + highCoefficientOffsetX) ];
  }

  sum += highFilter[0] * valsHigh[3];
  //sum += highFilter[1] * valsHigh[1];
  sum += highFilter[2] * valsHigh[2];
  //sum += highFilter[3] * valsHigh[3];
  sum += highFilter[4] * valsHigh[1];
  //sum += highFilter[5] * valsHigh[5];
  sum += highFilter[6] * valsHigh[0];
  //sum += highFilter[7] * valsHigh[7];
  //sum += highFilter[8] * valsHigh[8];

  int64 outputIndex = (yIndexLocal + inputImageMeta.yStart) * stride + (inputImageMeta.xStart + xIndexLocal);
  reconstructedSignal[outputIndex] = valsHigh[3];
  reconstructedSignal[outputIndex] = sum;
}

void iDwt2D(MyVector & L, int levelsToCompressUncompress,
            double * deviceInputSignal,
            struct ImageMeta & inputImageMeta,
            double * deviceILowFilter,
            double * deviceIHighFilter,
            int64 filterLength,
            double * deviceOutputCoefficients,
            double * deviceTmpMemory) {

  bool isHorizontal = true;
  //calculate current image meta
  struct ImageMeta currentImageMeta = inputImageMeta;

  for (int level = 0; level < levelsToCompressUncompress; level++) {

    std::cerr<<currentImageMeta.xEnd<<","<<currentImageMeta.yEnd<<std::endl;

    int64 totalNumElements = currentImageMeta.xEnd *  currentImageMeta.yEnd  * 4;
    int threads;
    dim3 blocks;
    calculateBlockSize(totalNumElements, threads, blocks);

    if (isHorizontal) {
      //inverseConvolveHorizontal <<< blocks, threads>>>(deviceTmpMemory, filterLength,
      inverseConvolveHorizontal <<< blocks, threads>>>(deviceInputSignal, filterLength,
          totalNumElements,
          deviceILowFilter, deviceIHighFilter,
          currentImageMeta,
          deviceTmpMemory);
    } else {
      inverseConvolveVertical <<< blocks, threads>>>(deviceTmpMemory, filterLength,
          totalNumElements,
          deviceILowFilter, deviceIHighFilter,
          currentImageMeta,
          deviceOutputCoefficients);
    }

    //if (isHorizontal) {
        //currentImageMeta.xEnd *= 2;
    //} else {
        //currentImageMeta.yEnd *= 2;
    //}

    isHorizontal = !isHorizontal;
  }

}
