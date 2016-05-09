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
void sobelFilter(Pixel *odata, int iw, int ih, float fScale, int blkSize) {
    CUDA_SAFE_CALL(cudaBindTextureToArray(tex, array));
#if 1 // Filter Enable Flag
    SobelTex<<<ih, blkSize>>>(odata, iw, iw, ih, fScale );
#else
    SobelCopyImage<<<ih, blkSize>>>(odata, iw, iw, ih );
#endif
    CUDA_SAFE_CALL(cudaUnbindTexture(tex));
}

