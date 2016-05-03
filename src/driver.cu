#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "sobel.h"

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define DIMENSION 512
#define INPUT_RAW inputData

// Exit true
bool testResult = true;

// Forward Declaration
void runTest(int argc, char **argv);

int main(int argc, char **argv) {

  runTest(argc, argv);
  cudaDeviceReset();

  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void runTest(int argc, char **argv) {
  int devID = findCudaDevice(argc, (const char **) argv);

  // Take input, if given
  int windowSize;
  const char *imageFilename = NULL;
  const char *outputFilename = NULL;
  if (argc == 4) {
    // Take Window size
    sscanf(argv[1], "%d", &windowSize);

    // Take Input file name
    imageFilename = argv[2];

    // Take Output file name
    outputFilename = argv[3];
  } else if (argc == 1){
    windowSize = 3;
    imageFilename = "lena.pgm";
    outputFilename = "lena_out.pgm";
  } else {
    printf("Usage: ./driver windowSize inputFile.pgm outputFile.pgm");
  }

  // load image from disk
  unsigned int width, height;
  char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

  if (imagePath == NULL) {
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  // Load image data to variable, store width and height
  Pixel *inputData = NULL;
  sdkLoadPGM(imagePath, &inputData, &width, &height);

  // Allocate size
  unsigned int size = width * height * sizeof(Pixel);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  // Copy input data to device
  // Pixel *hData = NULL;
  // checkCudaErrors(cudaMalloc((void **) &hData, size));

  // Allocate device memory for result
  Pixel *dData = NULL;
  checkCudaErrors(cudaMalloc((void **) &dData, size));

  // Copy image data from host to device
  // checkCudaErrors(cudaMemcpy(hData, INPUT_RAW, size, cudaMemcpyHostToDevice));
  setupTexture(width, height, INPUT_RAW);

  // Timing analysis loops
  short blockSizeHolder[] = {8, 16};

#if 0 // # Analysis Mode Flag
  // Loop blockSize: 8 or 16
  for(short blockSizeIndex = 0; blockSizeIndex < 2; blockSizeIndex++) {
    // Iterate each, 10 times
    for(short loop = 0; loop < 10; loop++) {
#else
#define blockSizeIndex (0)
{{
#endif

      // Specify block and grid dimensions
      dim3 dimBlock(blockSizeHolder[blockSizeIndex], blockSizeHolder[blockSizeIndex], 1);
      dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

      // Warmup
      // medianFilterKernel<<<dimGrid, dimBlock, 0>>>(hData, dData, width, height, windowSizeHolder[windowSizeIndex]);

      // Synchronize, and start timer
      checkCudaErrors(cudaDeviceSynchronize());
      StopWatchInterface *timer = NULL;
      sdkCreateTimer(&timer);
      sdkStartTimer(&timer);
      // Execute the kernel
      // medianFilterKernel<<<dimGrid, dimBlock, 0>>>(hData, dData, width, height, windowSizeHolder[windowSizeIndex]);
      sobelFilter(dData, width, height, 1.f);
      
      // Check if kernel execution generated an error
      getLastCudaError("Kernel execution failed");

      // Synchronize, and stop timer
      checkCudaErrors(cudaDeviceSynchronize());
      sdkStopTimer(&timer);
      printf("Processing time: %fms\n", sdkGetTimerValue(&timer));
      printf("%.2f Mpixels/sec\n",
             (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
      sdkDeleteTimer(&timer);

      // Allocate mem for the result on host side
      Pixel *hOutputData = (Pixel *) malloc(size);
      
      // copy result from device to host
      checkCudaErrors(cudaMemcpy(hOutputData, dData, size, cudaMemcpyDeviceToHost));
      
      // Write result to file
      sdkSavePGM(outputFilename, hOutputData, width, height);
      printf("Wrote '%s'\n", outputFilename);
    }

    // Ask Golden function to generate its output, compare and provide match info
    // char goldenFunction[2048];
    // sprintf(goldenFunction, "bin/standard %d lena.pgm > lena_out_gold.pgm", windowSizeHolder[windowSizeIndex]);
    // system(goldenFunction);
    // system("bin/diff lena_out_gold.pgm lena_out.pgm");      
  }

  // Free data
  checkCudaErrors(cudaFree(dData));
  free(imagePath);
  free(inputData);
}