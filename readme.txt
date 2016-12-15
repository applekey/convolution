-DSHARED_MEMORY for using shared memory optimization
-DBIG for inputs larger than 2048 when using shared memory

./wave 16384 5 2 0

##parms
16384, size of signal, for 2D case it is the width or height
5, levels of compression, for 2d, 1 level is both horizontal and vertical
2, test case, use 1 for 1d and 2 for 2d
0, 0 or 1, to either print the results of the decompose/reconstruct, 0 is for silent

