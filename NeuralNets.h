/*
	CSC D84 - Neural Networks

	This file contains the API function headers for your assignment.
	Please pay close attention to the function prototypes, and
	understand what the arguments are.

	Stubs for implementing each function are to be found in NeuralNets.c,
	along with clear ** TO DO markers to let you know where to add code.

	You are free to add helper functions within reason. But you must
	provide a prototype *in this file* as well as the implementation
	in the .c program file.

	Starter: (C) F. Estrada, Updated, Sep 2025
*/

#ifndef __NeuralNets_header

#define __NeuralNets_header
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<malloc.h>
#include<string.h>

#define CLASSES 10					// Number of classes (up to 10, bc. digits, CIFAR-10!)
#define SIDE 32						// Width of each image
#define SIZE 32*32					// Input image size
#define INPUTS (SIZE+1)			    // Number of inputs adjust as needed
#define OUTPUTS 10					// 1 per class, winner takes all
#define MAX_HIDDEN 25*25			// Maximum number of hidden units
#define ALPHA .01					// Network learning rate
#define SIGMOID_SCALE .01			// Scaling factor for sigmoid function input <--- MIND THIS!
#define PI 3.1415926535

int train_1layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS]);
int train_2layer_net(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS]);
void feedforward_1layer(double sample[INPUTS], double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS], double activations[OUTPUTS]);
void feedforward_2layer(double sample[INPUTS], double (*sigmoid)(double input), double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS], double h_activations[MAX_HIDDEN],double activations[OUTPUTS], int units);
void backprop_1layer(double sample[INPUTS],double activations[OUTPUTS], double (*sigmoid)(double input), int label, double weights_io[INPUTS][OUTPUTS]);
void backprop_2layer(double sample[INPUTS],double h_activations[MAX_HIDDEN], double activations[OUTPUTS], double (*sigmoid)(double input), int label, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS], int units);
int classify_1layer(double sample[INPUTS], int label, double (*sigmoid)(double input), double weights_io[INPUTS][OUTPUTS]);
int classify_2layer(double sample[INPUTS],int label,double (*sigmoid)(double input), int units, double weights_ih[INPUTS][MAX_HIDDEN], double weights_ho[MAX_HIDDEN][OUTPUTS]);
double logistic(double input);

#endif

