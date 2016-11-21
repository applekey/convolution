#include "helper.h"

struct ImageMeta {
    int64 imageWidth, imageHeight;
    int64 xStart, xEnd;
    int64 yStart, yEnd;
};

struct vec2 {
    int64 x, y;
};
void dwt2D_Horizontal(MyVector & L, int levelsToCompress,
                      double * deviceInputSignal, int64 signalLength,
                      double * deviceLowFilter,
                      double * deviceHighFilter,
                      double * deviceOutputCoefficients,
                      int64 filterLength) {


}
