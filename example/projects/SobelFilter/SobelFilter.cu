/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cutil.h>
#include <cuda_gl_interop.h>

#include "SobelFilter_kernels.cu"

//
// Cuda example code that implements the Sobel edge detection
// filter. This code works for 8-bit monochrome images.
//
// Use the '-' and '=' keys to change the scale factor.
//
// Other keys:
// I: display image
// T: display Sobel edge detection (computed solely with texture)
// S: display Sobel edge detection (computed with texture and shared memory)

void cleanup(void);
void initializeData(int w, int h);

static int wWidth   = 512; // Window width
static int wHeight  = 512; // Window height
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height

static int fpsCount = 0;        // FPS count for averaging
static int fpsLimit = 1;        // FPS limit for sampling
unsigned int timer;

// Display Data
static GLuint pbuffer = 0;     // Front and back CA buffers
static GLuint texid = 0;       // Texture for display
Pixel *pixels = NULL;          // Image pixel data on the host    
float imageScale = 1.f;        // Image exposure
enum SobelDisplayMode g_SobelDisplayMode;

static cudaArray *array = NULL;


#define OFFSET(i) ((char *)NULL + (i))

// Wrapper for the __global__ call that sets up the texture and threads
void 
sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode, 
            float fScale) {

    CUDA_SAFE_CALL(cudaBindTextureToArray(tex, array));

    switch ( mode ) {
        case  SOBELDISPLAY_IMAGE: 
            SobelCopyImage<<<ih, 384>>>(odata, iw, iw, ih );
            break;
        case SOBELDISPLAY_SOBELTEX:
            SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale );
            break;
        case SOBELDISPLAY_SOBELSHARED:
        {
            dim3 threads(16,4);
#ifndef FIXED_BLOCKWIDTH
	          int BlockWidth = 80; // must be divisible by 16 for coalescing
#endif
        		dim3 blocks = dim3(iw/(4*BlockWidth)+(0!=iw%(4*BlockWidth)),
                               ih/threads.y+(0!=ih%threads.y));
        		int SharedPitch = ~0x3f&(4*(BlockWidth+2*Radius)+0x3f);
        		int sharedMem = SharedPitch*(threads.y+2*Radius);

        		// for the shared kernel, width must be divisible by 4
        		iw &= ~3;

        		SobelShared<<<blocks, threads, sharedMem>>>((uchar4 *) odata, 
                                                        iw, 
#ifndef FIXED_BLOCKWIDTH
                                                        BlockWidth, SharedPitch,
#endif
                                                		    iw, ih, fScale );
        }
        break;
    }

    CUDA_SAFE_CALL(cudaUnbindTexture(tex));
}

void display(void) {  
   
   // Sobel operation
   Pixel *data = NULL;
   CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&data, pbuffer));
   CUT_SAFE_CALL(cutStartTimer(timer));  
   sobelFilter(data, imWidth, imHeight, g_SobelDisplayMode, imageScale );
   CUT_SAFE_CALL(cutStopTimer(timer));  
   CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbuffer));   
   
   glClear(GL_COLOR_BUFFER_BIT);

   glBindTexture(GL_TEXTURE_2D, texid);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbuffer);
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imWidth, imHeight, 
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, OFFSET(0));
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

   glDisable(GL_DEPTH_TEST);
   glEnable(GL_TEXTURE_2D);
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
   glBegin(GL_QUADS);
   glVertex2f(0, 0); glTexCoord2f(0, 0);
   glVertex2f(0, 1); glTexCoord2f(1, 0);
   glVertex2f(1, 1); glTexCoord2f(1, 1);
   glVertex2f(1, 0); glTexCoord2f(0, 1);
   glEnd();
   glBindTexture(GL_TEXTURE_2D, 0);

   glutSwapBuffers();
    
   fpsCount++;
   if (fpsCount == fpsLimit) {
       char fps[256];
       float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
       sprintf(fps, "Cuda Edge Detection: %3.1f fps", ifps);  
       glutSetWindowTitle(fps);
       fpsCount = 0; 
       fpsLimit = (int)max(ifps, 1.f);
       CUT_SAFE_CALL(cutResetTimer(timer));  
    }

    glutPostRedisplay();
}

void idle(void) {
    glutPostRedisplay();
}

void keyboard( unsigned char key, int x, int y) {
    switch( key) {
        case 27:
        exit (0);
		break;
		case '-':
			imageScale -= 0.1f;
		break;
		case '=':
			imageScale += 0.1f;
		break;
		case 'i': case 'I':
		    g_SobelDisplayMode = SOBELDISPLAY_IMAGE;
		break;
		case 's': case 'S':
		    g_SobelDisplayMode = SOBELDISPLAY_SOBELSHARED;
		break;
		case 't': case 'T':
		    g_SobelDisplayMode = SOBELDISPLAY_SOBELTEX;
        default: break;
    }
    glutPostRedisplay();
}

void reshape(int x, int y) {
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1); 
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

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

void cleanup(void) {
    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbuffer));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbuffer);
    glDeleteTextures(1, &texid);

    deleteTexture();

    CUT_SAFE_CALL(cutDeleteTimer(timer));  
}

void initializeData(char *file) {
    GLint bsize;
    unsigned int w, h;
    if (cutLoadPGMub(file, &pixels, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", file);
        exit(-1);
    }
    imWidth = (int)w; imHeight = (int)h;
    setupTexture(imWidth, imHeight, pixels);
    memset(pixels, 0x0, sizeof(Pixel) * imWidth * imHeight);
	
    glGenBuffers(1, &pbuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 
                    sizeof(Pixel) * imWidth * imHeight, 
                    pixels, GL_STREAM_DRAW);

    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize); 
    if (bsize != (sizeof(Pixel) * imWidth * imHeight)) {
        printf("Buffer object (%d) has incorrect size (%d).\n", pbuffer, bsize);
        exit(-1);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbuffer));

    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, imWidth, imHeight, 
                 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
}

void loadDefaultImage( char* loc_exec) {

    printf("Reading image lena.pgm.\n");
    const char* image_filename = "lena.pgm";
    char* image_path = cutFindFilePath(image_filename, loc_exec);
    if (image_path == 0) {
       printf( "Reading image failed.\n");
       exit(EXIT_FAILURE);
    }
    initializeData( image_path);
    cutFree( image_path);
}

int main(int argc, char** argv) {

    CUT_DEVICE_INIT();

    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutResetTimer(timer));  
 
    glutInit( &argc, argv);    
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("Cuda Edge Detection");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }

    if (argc > 1) {
        // test if in regression mode
        if( 0 != strcmp( "-noprompt", argv[1])) {
           initializeData(argv[1]);
        }
        else {
            loadDefaultImage( argv[0]);
        }
    } else {
        loadDefaultImage( argv[0]);
    }
    printf("I: display image\n");
    printf("T: display Sobel edge detection (computed with tex)\n");
    printf("S: display Sobel edge detection (computed with tex+shared memory)\n");
    printf("Use the '-' and '=' keys to change the brightness.\n");
    fflush(stdout);
    atexit(cleanup); 
    glutMainLoop();

    return 0;
}
