#ifndef __SOBELFILTER_KERNELS_H_
#define __SOBELFILTER_KERNELS_H_
#include <helper_cuda.h>

#define TILEX   36 // Sobel tile width
#define TILEY   28 // Sobel tile height
#define FTILEX  38 // Filter tile width
#define FTILEY  30 // Filter tile height
#define PTILEX  48 // TILEX + 2*RADIUS+1
#define PTILEY  40 // TILEY + 2*RADIUS+1 
#define TIDSX   48 // Threads in X
#define TIDSY   8  // Threads in Y
#define RADIUS  5  // Filter radius
#define RPIXELS  5  // Pixels per thread
#define WPIXELS  4  // Pixels per thread

typedef unsigned char Pixel;

// global determines which filter to invoke
// enum SobelDisplayMode {
//  SOBELDISPLAY_IMAGE = 0,
//  SOBELDISPLAY_SOBELTEX,
//  SOBELDISPLAY_SOBELSHARED
// };

// extern enum SobelDisplayMode g_SobelDisplayMode;

void setupTexture(int iw, int ih, Pixel *data);
void sobelFilter(Pixel *odata, int iw, int ih, float fScale);
void deleteTexture(void);

#endif

