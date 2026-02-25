/*
	CSC D84 - Neural Networks
	
    Open GL display and interface, data definitions for Neural Nets
    
    - THERE IS NOTHING YOU NEED TO MODIFY IN THIS FILE -

	Starter: (c) F. Estrada, Updated Sep. 2025
*/

#ifndef __NeuralNets_core_headers

#define __NeuralNets_core_headers

// Standard libraries
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <string.h>

// Open GL headers - must be installed on the system's include dirs.
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

// Function prototypes for student code
#include "NeuralNets.h"

#define IMAGE_OUTPUT 0		// Enable to output each frame to disk
#define USE_GABOR 0			// Set to 1 to use a gabor filter as initial kernel for 2-layer NN

//************  USEFUL DATA STRUCTURES **********************

// ***********  GLOBAL DATA  ********************************
unsigned char false_pos[390][390];				// Image for false-positive examples
unsigned char false_neg[390][390];				// Image for false-negatuve examples
unsigned char network_weights[390][390];		// Image for network weight samples
unsigned char fb[1024*1024*3];					// OpenGL framebuffer
double kernelC[SIDE*SIDE];
double kernelS[SIDE*SIDE];
double w_io[INPUTS][OUTPUTS];					// Weights from input to output, 1-layer Networks
double w_ih[INPUTS][MAX_HIDDEN];				// Weights from input to hidden layer, 2-layer Networks
double w_ho[MAX_HIDDEN][OUTPUTS];				// Weights from hidden to output, 2-layer Network
int r_seed;										// Initial random seed
int mode; 										// Search mode
int sigm;										// Selected sigmoid function
int units;										// Number of hidden units
int n_train=40000;								// Number of training images
int n_test=20000;								// Number of testing images
unsigned char *trainingDigits;					// Training digits data
unsigned char *trainingLabels;					// Training labels data
unsigned char *testingDigits;					// Testing digits data
unsigned char *testingLabels;					// Testing labels data
int DATASET=0;									// 0 - MNIST(32), 1 - CIFAR-10
int doneTrain=0;								// Flag to determine if training should continue
int activeFB=10;								// Active frame buffer

/* Had to move here from main() because we had to move code from main() to
 * the OpenGL loop */
int windowID;										
int count_iters=0;
int correct_class_counter[CLASSES];
int total_class_counter[CLASSES];
int fp_counter[10],fn_counter[10];
double sample[INPUTS];
double patch[SIDE*SIDE];
double w2l[INPUTS][10];
int idx,cls,ix,iy;
int no_improv=0;
int max_iters;
double mi,mx;
double new_avg=0;
double best_avg=0;

// ***********  FUNCTION HEADER DECLARATIONS ****************

// Image processing functions
void imageOutput(unsigned char *im, int sx, int sy, const char *filename);
void imageConvert(unsigned char *grayIm, int sx, int sy, const char *filename);
void gabor(double sig, double f, double th, int ox, int oy, double *kernelC, double *kernelS);
void renderHiddenResponse(int c,double (*sigmoid)(double input));
void renderOutputLayerWeights();
void renderFPFN();

// Open GL callbacks for handling events in glut
void initGlut(const char* winName, int sizeX, int sizeY, int positionX, int positionY);
void WindowReshape(int w, int h);
void WindowDisplay(void);
void kbHandler(unsigned char key, int x, int y);
#endif

