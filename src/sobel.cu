#include <stdio.h>
#include <stdlib.h>
#include "sobel.h"

#define SV 0.003921f
#define IV 255.f

// Texture reference for reading image
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];

#define Radius 1

#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

#define ABS(x) ((x)<0?-(x):(x))

__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale )
{
    short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
    short Vert = ul + 2*um + ur - ll - 2*lm - lr;
    short Sum = (short) (fScale*(ABS(Horz)+ABS(Vert)));
    if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
    return (unsigned char) Sum;
}

// __global__ void 
// SobelShared( uchar4 *pSobelOriginal, unsigned short SobelPitch, 
// #ifndef FIXED_BLOCKWIDTH
//              short BlockWidth, short SharedPitch,
// #endif
//              short w, short h, float fScale )
// { 
//     short u = 4*blockIdx.x*BlockWidth;
//     short v = blockIdx.y*blockDim.y + threadIdx.y;
//     short ib;

//     int SharedIdx = threadIdx.y * SharedPitch;

//     for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
//         LocalBlock[SharedIdx+4*ib+0] = tex2D( tex, 
//             (float) (u+4*ib-Radius+0), (float) (v-Radius) );
//         LocalBlock[SharedIdx+4*ib+1] = tex2D( tex, 
//             (float) (u+4*ib-Radius+1), (float) (v-Radius) );
//         LocalBlock[SharedIdx+4*ib+2] = tex2D( tex, 
//             (float) (u+4*ib-Radius+2), (float) (v-Radius) );
//         LocalBlock[SharedIdx+4*ib+3] = tex2D( tex, 
//             (float) (u+4*ib-Radius+3), (float) (v-Radius) );
//     }
//     if ( threadIdx.y < Radius*2 ) {
//         //
//         // copy trailing Radius*2 rows of pixels into shared
//         //
//         SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
//         for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
//             LocalBlock[SharedIdx+4*ib+0] = tex2D( tex, 
//                 (float) (u+4*ib-Radius+0), (float) (v+blockDim.y-Radius) );
//             LocalBlock[SharedIdx+4*ib+1] = tex2D( tex, 
//                 (float) (u+4*ib-Radius+1), (float) (v+blockDim.y-Radius) );
//             LocalBlock[SharedIdx+4*ib+2] = tex2D( tex, 
//                 (float) (u+4*ib-Radius+2), (float) (v+blockDim.y-Radius) );
//             LocalBlock[SharedIdx+4*ib+3] = tex2D( tex, 
//                 (float) (u+4*ib-Radius+3), (float) (v+blockDim.y-Radius) );
//         }
//     }

//     __syncthreads();

//     u >>= 2;    // index as uchar4 from here
//     uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
//     SharedIdx = threadIdx.y * SharedPitch;

//     for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

//         unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
//         unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
//         unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
//         unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
//         unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
//         unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
//         unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
//         unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
//         unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

//         uchar4 out;

//         out.x = ComputeSobel(pix00, pix01, pix02, 
//                              pix10, pix11, pix12, 
//                              pix20, pix21, pix22, fScale );

//         pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
//         pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
//         pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
//         out.y = ComputeSobel(pix01, pix02, pix00, 
//                              pix11, pix12, pix10, 
//                              pix21, pix22, pix20, fScale );

//         pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
//         pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
//         pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
//         out.z = ComputeSobel( pix02, pix00, pix01, 
//                               pix12, pix10, pix11, 
//                               pix22, pix20, pix21, fScale );

//         pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
//         pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
//         pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
//         out.w = ComputeSobel( pix00, pix01, pix02, 
//                               pix10, pix11, pix12, 
//                               pix20, pix21, pix22, fScale );
//         if ( u+ib < w/4 && v < h ) {
//             pSobel[u+ib] = out;
//         }
//     }

//     __syncthreads();
// }

__global__ void 
SobelCopyImage( unsigned char *pSobelOriginal, unsigned int Pitch, 
                int w, int h )
{ 
    unsigned char *pSobel = 
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        pSobel[i] = tex2D( tex, (float) i, (float) blockIdx.x );
    }
}

__global__ void 
SobelTex( unsigned char *pSobelOriginal, unsigned int Pitch, 
          int w, int h, float fScale )
{ 
    unsigned char *pSobel = 
      (unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
    for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
        unsigned char pix00 = tex2D( tex, (float) i-1, (float) blockIdx.x-1 );
        unsigned char pix01 = tex2D( tex, (float) i+0, (float) blockIdx.x-1 );
        unsigned char pix02 = tex2D( tex, (float) i+1, (float) blockIdx.x-1 );
        unsigned char pix10 = tex2D( tex, (float) i-1, (float) blockIdx.x+0 );
        unsigned char pix11 = tex2D( tex, (float) i+0, (float) blockIdx.x+0 );
        unsigned char pix12 = tex2D( tex, (float) i+1, (float) blockIdx.x+0 );
        unsigned char pix20 = tex2D( tex, (float) i-1, (float) blockIdx.x+1 );
        unsigned char pix21 = tex2D( tex, (float) i+0, (float) blockIdx.x+1 );
        unsigned char pix22 = tex2D( tex, (float) i+1, (float) blockIdx.x+1 );
        pSobel[i] = ComputeSobel(pix00, pix01, pix02, 
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22, fScale );
    }
}

#define CUDA_SAFE_CALL checkCudaErrors
static cudaArray *array = NULL;
void setupTexture(int iw, int ih, Pixel *data)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    CUDA_SAFE_CALL(cudaMallocArray(&array, &desc, iw, ih));
    CUDA_SAFE_CALL(cudaMemcpyToArray(array, 0, 0, data, sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice));
}

void deleteTexture(void)
{
    CUDA_SAFE_CALL(cudaFreeArray(array));
}

//float *inputData, float *outputData, int width, int height, int filterSize
void sobelFilter(Pixel *odata, int iw, int ih, float fScale) {
    CUDA_SAFE_CALL(cudaBindTextureToArray(tex, array));
#if 0 // Filter Enable Flag
    SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale );
#else
    SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih );
#endif
    CUDA_SAFE_CALL(cudaUnbindTexture(tex));
}

