/*
  CSC D84 - Neural Networks

  This file contains the driver for the Neural Networks code. 
  
  What goes here:
	* OpenGL init and window management (can be safely ignored)
	* Image rendering - Maze generation and overlay of game sprites
	* Main training loop - corresponding to the GLUT main loop
	* Evaluation code - which runs on trained neural nets
	
  ** There is NOTHING you need to modify or add in here **

  (c) F. Estrada, Updated Sep. 2025
*/

#include "NeuralNets_core.h"
#include "DataLoad.c"             // Which will include DataLoad.h
                                  // and create the arrays for CIFAR-10 data

int main(int argc, char *argv[])
{
 char line[1024];
 FILE *f;
 
 // Command line parsing
 if (argc<6)
 {
  fprintf(stderr,"Incorrect usage\n");
  fprintf(stderr,"  NeuralNets mode units sigmoid max_iters dataset\n");
  fprintf(stderr,"   mode:\n");
  fprintf(stderr,"         0 - single layer network training\n");
  fprintf(stderr,"         1 - single layer network testing\n");
  fprintf(stderr,"         2 - two-layer network training\n");
  fprintf(stderr,"         3 - two-layer network testing\n");
  fprintf(stderr,"   units: Number of hidden-layer units for mode 1 and 3 in [2 - %d]\n",MAX_HIDDEN);
  fprintf(stderr,"   sigmoid: 0 - use Logistic function, 1 - use Hyperbolic Tangent\n");
  fprintf(stderr,"   max_iters: [10 - 250] max number of iterations with no improvement before stopping\n");
  fprintf(stderr,"   dataset: 0 -> MNIST, 1-> CIFAR-10\n");
  exit(0);
 }
 mode=atoi(argv[1]);
 units=atoi(argv[2]);
 sigm=atoi(argv[3]);
 max_iters=atoi(argv[4])*5;
 DATASET=atoi(argv[5]);
 n_test=60000-n_train;
 
 if (mode>3||mode<0||sigm<0||sigm>1||(mode>1&&(units<2||units>MAX_HIDDEN))||max_iters<10||max_iters>2500||DATASET<0||DATASET>1)
 {
  fprintf(stderr,"Incorrect usage\n");
  fprintf(stderr,"  NeuralNets mode units sigmoid max_iters dataset\n");
  fprintf(stderr,"   mode:\n");
  fprintf(stderr,"         0 - single layer network training\n");
  fprintf(stderr,"         1 - single layer network testing\n");
  fprintf(stderr,"         2 - two-layer network training\n");
  fprintf(stderr,"         3 - two-layer network testing\n");
  fprintf(stderr,"   units: Number of hidden-layer units for mode 1 and 3 in [2 - %d]\n",MAX_HIDDEN);
  fprintf(stderr,"   sigmoid: 0 - use Logistic function, 1 - use Hyperbolic Tangent\n");
  fprintf(stderr,"   max_iters: [10 - 250] max number of iterations with no improvement before stopping\n");
  fprintf(stderr,"   max_iters: [10 - 250] max number of iterations with no improvement before stopping\n");
  exit(0);
 }

 if (DATASET==1)
 {
  // Read CIFAR 10 data
  if (!readCIFAR10())
  {
    fprintf(stderr,"Unable to load image data - please check data files are in the same directory as the program.\n");
    exit(0);
  }  
  trainingDigits=(unsigned char *)calloc(n_train*SIZE,sizeof(unsigned char));       
  trainingLabels=(unsigned char *)calloc(n_train,sizeof(unsigned char));
  testingDigits=(unsigned char *)calloc(n_test*SIZE,sizeof(unsigned char));
  testingLabels=(unsigned char *)calloc(n_test,sizeof(unsigned char));
  if (testingDigits==NULL||testingLabels==NULL||trainingDigits==NULL||trainingLabels==NULL)
  {
   fprintf(stderr,"Unable to allocate memory for training/testing data\n");
   exit(0);
  }
  sampleSplit(n_train,n_test);
 }
 else
 {
   n_train=60000;
   n_test=10000;
   trainingDigits=(unsigned char *)calloc(n_train*SIZE,sizeof(unsigned char));       
   trainingLabels=(unsigned char *)calloc(n_train,sizeof(unsigned char));
   testingDigits=(unsigned char *)calloc(n_test*SIZE,sizeof(unsigned char));
   testingLabels=(unsigned char *)calloc(n_test,sizeof(unsigned char));
   if (testingDigits==NULL||testingLabels==NULL||trainingDigits==NULL||trainingLabels==NULL)
   {
    fprintf(stderr,"Unable to allocate memory for training/testing data\n");
    exit(0);
   }
   f=fopen("training-labels.dat","r");
   if (f==NULL)
   {
     fprintf(stderr,"Unable to load training labels for digits.\n");
     free(trainingDigits);
     free(trainingLabels);
     free(testingDigits);
     free(testingLabels);
     exit(0);
   }
   fread(trainingLabels,60000*sizeof(unsigned char),1,f);
   fclose(f);
   f=fopen("training-digits32.dat","r");
   if (f==NULL)
   {
     fprintf(stderr,"Unable to load training images for digits.\n");
     free(trainingDigits);
     free(trainingLabels);
     free(testingDigits);
     free(testingLabels);
     exit(0);
   }
   fread(trainingDigits,32*32*60000*sizeof(unsigned char),1,f);
   fclose(f);
   f=fopen("testing-labels.dat","r");
   if (f==NULL)
   {
     fprintf(stderr,"Unable to load testing labels for digits.\n");
     free(trainingDigits);
     free(trainingLabels);
     free(testingDigits);
     free(testingLabels);
     exit(0);
   }
   fread(testingLabels,10000*sizeof(unsigned char),1,f);
   fclose(f);
   f=fopen("testing-digits32.dat","r");
   if (f==NULL)
   {
     fprintf(stderr,"Unable to load testing images for digits.\n");
     free(trainingDigits);
     free(trainingLabels);
     free(testingDigits);
     free(testingLabels);
     exit(0);
   }
   fread(testingDigits,32*32*10000*sizeof(unsigned char),1,f);
   fclose(f);   
 }
 
 // Initialize network weights	
 for (int i=0; i<INPUTS; i++)
 {
  for (int j=0; j<OUTPUTS; j++)
   w_io[i][j]=(drand48()-.5)*.1;
  for (int j=0; j<units; j++)
  {
   if (!USE_GABOR) w_ih[i][j]=drand48()-.5;
        
   for (int k=0; k<OUTPUTS; k++)
    w_ho[j][k]=(drand48()-.5)*.1;
  }   
 }
 if (USE_GABOR)
 {
  for (int j=0; j<units; j++)          // Init using Gabor filters
  {
    gabor(2.5*(drand48()+.5),4+(5*drand48()),drand48()*2.0*PI,(SIDE/3)+(int)(SIDE*.66*drand48()),(SIDE/3)+(int)(SIDE*.66*drand48()),&kernelC[0],&kernelS[0]);
    if (drand48()<=.5)
    {
      for (int ii=0; ii<INPUTS-1; ii++)
        w_ih[ii][j]=kernelC[ii];
    }
    else
    {
      for (int ii=0; ii<INPUTS-1; ii++)
        w_ih[ii][j]=kernelS[ii];
    }
  }
 }
 
 // Initialize output images
 memset(&fb[0],0,1024*1024*3*sizeof(unsigned char));
 memset(&false_pos[0][0],0,390*390*sizeof(unsigned char));
 memset(&false_neg[0][0],0,390*390*sizeof(unsigned char));
 memset(&network_weights[0][0],0,390*390*sizeof(unsigned char));
 
 // Intialize GLUT and OpenGL, and launch the window display loop
 glutInit(&argc, argv);
 initGlut("Neural Nets Classifier",1024,1024,10,10);
 glutMainLoop();   
 
 exit(0);   // <--- this is never reached  
}

void imageConvert(unsigned char *grayIm, int sx, int sy, const char *filename)
{
 // Convert a grayscale image to RGB for output
 unsigned char *rgbIm;
 rgbIm=(unsigned char *)calloc(sx*sy*3,sizeof(unsigned char));
 if (rgbIm==NULL)
 {
  fprintf(stderr,"Unable to allocate memory for image output!\n");
  return;
 }
 
 for (int i=0; i<sx; i++)
   for (int j=0; j<sy; j++)
   {
    *(rgbIm+((i+(j*sx))*3)+0)=*(grayIm+i+(j*sx));
    *(rgbIm+((i+(j*sx))*3)+1)=*(grayIm+i+(j*sx));
    *(rgbIm+((i+(j*sx))*3)+2)=*(grayIm+i+(j*sx));
   }
 imageOutput(rgbIm,sx,sy,filename);
 free(rgbIm);
}

void imageOutput(unsigned char *im, int sx, int sy, const char *filename)
{
 // Writes out a .ppm file from the image data contained in 'im'.
 // Note that Windows typically doesn't know how to open .ppm
 // images. Use Gimp or any other seious image processing
 // software to display .ppm images.
 // Also, note that because of Windows file format management,
 // you may have to modify this file to get image output on
 // Windows machines to work properly.
 //
 // Assumes a 24 bit per pixel image stored as unsigned chars
 //

 FILE *f;

 if (im!=NULL)
  {
   f=fopen(filename,"wb+");
   if (f==NULL)
   {
    fprintf(stderr,"Unable to open file %s for output! No image written\n",filename);
    return;
   }
   fprintf(f,"P6\n");
   fprintf(f,"# Generated by code\n");
   fprintf(f,"%d %d\n",sx,sy);
   fprintf(f,"255\n");
   fwrite((unsigned char *)im,sx*sy*3*sizeof(unsigned char),1,f);
   fclose(f);
   return;
  }
 fprintf(stderr,"imageOutput(): Specified image is empty. Nothing output\n");
}

void renderFPFN()
{
  // Render false-positive and false-negative images to the OpenGL 
  // frame buffer
  int ix, iy;
  
  // False negatives (to the left)
  ix=112;
  iy=312;
  for (int j=0; j<390; j++)
    for (int i=0; i<390; i++)
    {
      *(fb+((ix+i+((iy+j)*1024))*3)+0)=false_neg[j][i];
      *(fb+((ix+i+((iy+j)*1024))*3)+1)=false_neg[j][i];
      *(fb+((ix+i+((iy+j)*1024))*3)+2)=false_neg[j][i];
    }
    
  // False positives (to the right)
  ix=527;
  iy=312;
  for (int j=0; j<390; j++)
    for (int i=0; i<390; i++)
    {
      *(fb+((ix+i+((iy+j)*1024))*3)+0)=false_pos[j][i];
      *(fb+((ix+i+((iy+j)*1024))*3)+1)=false_pos[j][i];
      *(fb+((ix+i+((iy+j)*1024))*3)+2)=false_pos[j][i];
    }
        
}

void renderOutputLayerWeights()
{
  // Renders an array of 5x2 images corresponding to the output layer
  // weights. This is intended for the 1-layer networks
  
  int ix,iy;
  double mx,mi;
  unsigned char patch[SIZE*3];
  unsigned char patch5[SIZE*25*3];
  double tmp[SIZE];
  double R1,G1,B1,R2,G2,B2,R3,G3,B3;
  FILE *f;
  char line[1024];

  R1=0;
  G1=(.55*255.0);
  B1=(.95*255.0);
    
  R2=(.99*255.0);
  G2=(1.0*255.0);
  B2=(.72*255.0);
    
  R3=(.23*255.0);
  G3=(.05*255.0);
  B3=(.45*255.0);

  memset(&fb[0],0,1024*1024*3*sizeof(unsigned char));    

  for (int i=0; i<10; i++)
  {         
    // Render the weights into the frame buffer at a size of 5x the original 
    memset(&patch[0],0,SIZE*sizeof(unsigned char));
    
    mi=10000;
    mx=-10000;
    for (int j=0; j<SIZE; j++)
    {
      if (w_io[j][i]>mx) mx=w_io[j][i];
      if (w_io[j][i]<mi) mi=w_io[j][i];      
    }
    for (int j=0; j<SIZE; j++)
    {
      tmp[j]=(w_io[j][i]-mi)/(mx-mi);
      patch[j*3]=(unsigned char)((1.0-tmp[j])*R3 + (tmp[j]*R2));
      patch[j*3+1]=(unsigned char)((1.0-tmp[j])*G3 + (tmp[j]*G2));
      patch[j*3+2]=(unsigned char)((1.0-tmp[j])*B3 + (tmp[j]*B2));
    }
    
    for (int j=0; j<SIDE*5; j++)
      for (int k=0; k<SIDE*5; k++)
      {
        ix=k/5;
        iy=j/5;
        *(patch5+((k+(j*(SIDE*5)))*3)+0)=*(patch+((ix+(iy*SIDE))*3)+0);
        *(patch5+((k+(j*(SIDE*5)))*3)+1)=*(patch+((ix+(iy*SIDE))*3)+1);
        *(patch5+((k+(j*(SIDE*5)))*3)+2)=*(patch+((ix+(iy*SIDE))*3)+2);
      }
      
    ix=37+((i%5)*5*38);
    iy=322+((i/5)*5*38);
    
    // Got a nice RGB image for this kernel, render to FB
    for (int yy=0; yy<SIDE*5; yy++)
      for (int xx=0; xx<SIDE*5; xx++)
      {
        *(fb+(((ix+xx)+((iy+yy)*1024))*3)+0)=patch5[(xx+(yy*SIDE*5))*3 + 0];
        *(fb+(((ix+xx)+((iy+yy)*1024))*3)+1)=patch5[(xx+(yy*SIDE*5))*3 + 1];
        *(fb+(((ix+xx)+((iy+yy)*1024))*3)+2)=patch5[(xx+(yy*SIDE*5))*3 + 2];
      }    
  }

  sprintf(&line[0],"output_kernels.ppm");
  f=fopen(line,"w");
  fprintf(f,"P6\n");
  fprintf(f,"# Input to ouput kernels\n");
  fprintf(f,"1024 1024\n");
  fprintf(f,"255\n");
  fwrite(fb,1024*1024*3*sizeof(unsigned char),1,f);
  fclose(f);
  
}

void renderHiddenResponse(int c,double (*sigmoid)(double input))
{
    // Renders an array of 12 x 12 images at size 64x64 showing the
    // response from the hidden layer to an input image in class c
    // This is to try to illustrate what the output neurons 'see'
    //
    // The rendered image shows the input to the output neuron,
    // so it is the weighted sum of hidden layer responses.

    int ix,iy;
    double acc[SIZE];
    double mx,mi;
    unsigned char patch[SIZE*3];
    int ds_idx=0;
    double R1,G1,B1,R2,G2,B2,R3,G3,B3;
    double acts[units];             // Man, it's nice to have a new compiler!
    FILE *f;
    char line[1024];
    int topN=15;
    
    R1=0;
    G1=(.55*255.0);
    B1=(.95*255.0);
    
    R2=(.99*255.0);
    G2=(1.0*255.0);
    B2=(.72*255.0);
    
    R3=(.23*255.0);
    G3=(.05*255.0);
    B3=(.45*255.0);
    
    memset(&fb[0],0,1024*1024*3*sizeof(unsigned char));    
    
    for (int i=0; i<12; i++)
      for (int j=0; j<12; j++)
      {
          ix=10+8+(i*84);
          iy=10+8+(j*84);
          
          memset(&acc[0],0,SIZE*sizeof(double));
          memset(&patch[0],0,SIZE*sizeof(unsigned char));
          
          while(*(trainingLabels+ds_idx)!=c) ds_idx++;
          
          // Compute the activations for each hidden layer neuron from this input image,
          // need those values to properly accumulate the kernel contributions
          memset(&acts[0],0,units*sizeof(double));
          for (int ii=0; ii<INPUTS; ii++)
            for (int jj=0; jj<units; jj++)
              acts[jj]+=*(trainingDigits+(ds_idx*SIZE)+ii)*w_ih[ii][jj];    
          for (int ii=0; ii<units; ii++) acts[ii]=sigmoid(SIGMOID_SCALE*acts[ii]);
          
          for (int k=0; k<topN; k++)      // Add up the top N responses
          { 
            mx=-10000;
            mi=-1;
            for (int kk=0; kk<units; kk++)
              if (fabs(acts[kk])>mx)
              {
                mx=acts[kk];
                mi=kk;
              }
            
            acts[(int)mi]=-10000;
            for (int l=0; l<SIZE; l++)
              if (j/6<1)          // Plot top halv vs. bottom half
               acc[l]+=*(trainingDigits+(ds_idx*SIZE)+l)*w_ih[l][(int)mi]*acts[(int)mi]*w_ho[(int)mi][c];
              else
               acc[l]+=*(trainingDigits+(ds_idx*SIZE)+l);
          }
          
          mx=-10000;
          mi=10000;
          for (int k=0; k<SIZE; k++)
          {
            if (acc[k]>mx) mx=acc[k];
            if (acc[k]<mi) mi=acc[k];
          }
          for (int k=0; k<SIZE; k++)
          {
            acc[k]=(acc[k]-mi)/(mx-mi);
            if (j/6<1)          // Plot top halv vs. bottom half
            {
             patch[k*3]=(unsigned char)((1.0-acc[k])*R3 + (acc[k]*R2));
             patch[k*3+1]=(unsigned char)((1.0-acc[k])*G3 + (acc[k]*G2));
             patch[k*3+2]=(unsigned char)((1.0-acc[k])*B3 + (acc[k]*B2));
            }
            else
            {
             patch[k*3]=(unsigned char)(255.0*acc[k]);
             patch[k*3+1]=(unsigned char)(255.0*acc[k]);
             patch[k*3+2]=(unsigned char)(255.0*acc[k]);
            }
          }
                    
          // Render to the framebuffer
          for (int yy=0; yy<SIDE; yy++)
            for (int xx=0; xx<SIDE; xx++)
            {
              *(fb+(((ix+2*xx)+((iy+2*yy)*1024))*3)+0)=patch[(xx+(yy*SIDE))*3 + 0];
              *(fb+(((ix+2*xx)+((iy+2*yy)*1024))*3)+1)=patch[(xx+(yy*SIDE))*3 + 1];
              *(fb+(((ix+2*xx)+((iy+2*yy)*1024))*3)+2)=patch[(xx+(yy*SIDE))*3 + 2];
              *(fb+(((ix+2*xx+1)+((iy+2*yy)*1024))*3)+0)=patch[(xx+(yy*SIDE))*3 + 0];
              *(fb+(((ix+2*xx+1)+((iy+2*yy)*1024))*3)+1)=patch[(xx+(yy*SIDE))*3 + 1];
              *(fb+(((ix+2*xx+1)+((iy+2*yy)*1024))*3)+2)=patch[(xx+(yy*SIDE))*3 + 2];
              *(fb+(((ix+2*xx)+((iy+2*yy+1)*1024))*3)+0)=patch[(xx+(yy*SIDE))*3 + 0];
              *(fb+(((ix+2*xx)+((iy+2*yy+1)*1024))*3)+1)=patch[(xx+(yy*SIDE))*3 + 1];
              *(fb+(((ix+2*xx)+((iy+2*yy+1)*1024))*3)+2)=patch[(xx+(yy*SIDE))*3 + 2];
              *(fb+(((ix+2*xx+1)+((iy+2*yy+1)*1024))*3)+0)=patch[(xx+(yy*SIDE))*3 + 0];
              *(fb+(((ix+2*xx+1)+((iy+2*yy+1)*1024))*3)+1)=patch[(xx+(yy*SIDE))*3 + 1];
              *(fb+(((ix+2*xx+1)+((iy+2*yy+1)*1024))*3)+2)=patch[(xx+(yy*SIDE))*3 + 2];
            }

          ds_idx++;
      }

      sprintf(&line[0],"responses_class_%d.ppm",c);
      f=fopen(line,"w");
      fprintf(f,"P6\n");
      fprintf(f,"# Responses to class %d from hidden layer features\n",c);
      fprintf(f,"1024 1024\n");
      fprintf(f,"255\n");
      fwrite(fb,1024*1024*3*sizeof(unsigned char),1,f);
      fclose(f);
}

void gabor(double sig, double f, double th, int ox, int oy, double *kernelC, double *kernelS)
{
  // Greate a gabor filter pair of size SIDExSIDE with the specified
  // sigma, frequency, and angle. Stores it in the arrays
  // provided via the kernel pointers for sine and cosine versions.
  
  int hs=SIDE/2;
  int i1,j1;
  double t;
  unsigned char tmp[SIDE*SIDE];
  FILE *f2;
  
  for (int j=0; j<SIDE; j++)
    for (int i=0; i<SIDE; i++)
    {
      i1=i-ox;
      j1=j-oy;
      *(kernelC+i+(j*SIDE))=exp(-((i1*i1)+(j1*j1))/(2*sig*sig))*cos(2.0*PI*f*((i*cos(th))+(j*sin(th))));
      *(kernelS+i+(j*SIDE))=exp(-((i1*i1)+(j1*j1))/(2*sig*sig))*sin(2.0*PI*f*((i*cos(th))+(j*sin(th))));
    }
}

// OpenGL stuff below - including the main display loop
void initGlut(const char* winName, int sizeX, int sizeY, int positionX, int positionY)
{
 // This is the GLUT library initialization funtion. GLUT provides a simple
 // API for initializing OpenGL and setting up a window on screen with
 // specified properties.
 //
 // Input arguments: 
 //    winName - Name of the OpenGL window being created (displayed on title bar)
 //    sizeX, sizeY - Window size in pixels
 //    positionX, positionY - Position of the window on the screen

 // Set video mode: double-buffered, color, depth-buffered
 glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

 // Create window
 glutInitWindowSize(sizeX,sizeY);
 glutInitWindowPosition (positionX,positionY);
 windowID = glutCreateWindow(winName);

 // Setup callback functions to handle window-related events.
 // In particular, OpenGL has to be informed of which functions
 // to call when the image needs to be refreshed, and when the
 // image window is being resized.
 glutReshapeFunc(WindowReshape);   	    // Call WindowReshape() whenever window resized
 glutDisplayFunc(WindowDisplay);   	    // Call WindowDisplay() whenever new frame needed
 glutKeyboardFunc(kbHandler);           // Keyboard handler function
}

void kbHandler(unsigned char key, int x, int y)
{
 // Expand this function to handle any keyboard interaction - this is called anytime a
 // key is pressed with the OpenGL window active.
 if (key>='0'&&key<='9')          // Display corresponding class for 2-layer net
   activeFB=key-0x30;
 if (key=='-') activeFB=10;       // Display network weights, 1 and 2 layer net
 if (key=='=') activeFB=11;       // Display false positives and false negatives
 if (key=='q') 
 {
  // Add here any code required to clean up before exit
  free(trainingDigits);
  free(trainingLabels);
  free(testingDigits);
  free(testingLabels); 
  exit(0);
 }
}

void WindowReshape(int width, int height)
{
 /*
   This function is called whenever the window is resized. It takes care of setting up
   OpenGL world-to-pixel-coordinate conversion matrices so that the image content is
   properly displayed regardless of window size.

   The width and height are provided by OpenGL's GLUT library upon the window being resized
 */

 // Setup projection matrix for new window
 glMatrixMode(GL_PROJECTION);
 glLoadIdentity();
 gluOrtho2D(0, 800, 800, 0);
 glViewport(0,0,width,height);
 glutPostRedisplay();
}

void WindowDisplay(void)
{
 /*
    This function is called whenever the frame needs to be updated. That means
    once per each iteration of the glutMainLoop, as well as anytime that the
    window is resized.

	Unfortunately, GLUT's display function prototype allows no arguments! that means
	any data initialized by main() and needed for the game has to be made global
    	or we would not be able to use it here. Apologies to Anya and Marcelo.
 */
 char line[1024];
 static int frame=0;
 static GLuint texture;
 static double scale_factor=1.0;
 FILE *f;

 /*
  *   Add here any code required to update the display - pretty much this means
  *  the code that actuall runs the program processing, and updates an image
  *  that is displayed on the OpenGL window as a 2D texture
  * 
  *   After the update is done, the code below will display the image on the
  *  OpenGL window.
  * 
  *   Note that the code below expects a 1024x1024 image, update as needed!
  */

  
 // Let's get something going
 if (doneTrain)
 {
   // Just display!
   memset(fb,0,1024*1024*3*sizeof(unsigned char));
   
   if (mode==0||mode==1)
   {
    if (activeFB<=10)
     renderOutputLayerWeights();
    else
     renderFPFN();
   }
   else
   {
    if (activeFB==10)
    {
     for (int i=0; i<units; i++)
     {
        mx=-10000;
        mi=10000;
        for (int j=0; j<INPUTS-1; j++)
        {
          patch[j]=w_ih[j][i];
          if (patch[j]>mx) mx=patch[j];
          if (patch[j]<mi) mi=patch[j];
        }
        for (int j=0; j<INPUTS-1; j++)
          if (mx-mi!=0) patch[j]=(patch[j]-mi)/(mx-mi);
        
        iy=37+((i/25)*38);
        ix=37+((i%25)*38);
                
        for (int k=0; k<SIDE; k++)
          for (int j=0; j<SIDE; j++) 
          {
              *(fb+(((ix+j)+((iy+k)*1024))*3)+0)=(unsigned char)(255*patch[j+(k*SIDE)]);
              *(fb+(((ix+j)+((iy+k)*1024))*3)+1)=(unsigned char)(255*patch[j+(k*SIDE)]);
              *(fb+(((ix+j)+((iy+k)*1024))*3)+2)=(unsigned char)(255*patch[j+(k*SIDE)]);
          }
     }
    }
    else if (activeFB>=0&&activeFB<10)
    {
      if (sigm==0) renderHiddenResponse(activeFB,logistic);
      else renderHiddenResponse(activeFB,tanh);
    }
    else renderFPFN();
   }
 }
 if ((mode==0||mode==2)&&!doneTrain)
 {
  memset(&correct_class_counter[0],0,CLASSES*sizeof(int));
  memset(&total_class_counter[0],0,CLASSES*sizeof(int));
  memset(&fp_counter[0],0,10*sizeof(int));
  memset(&fn_counter[0],0,10*sizeof(int));  
  memset(fb,0,1024*1024*3*sizeof(unsigned char));

  // Neural net training round
  while (count_iters<2000)
  {
   // Randomly choose next sample from the training set
   idx=(int)((double)n_train*drand48());
   for (int i=0; i<INPUTS-1; i++)
    sample[i]=((double)(*(trainingDigits+((INPUTS-1)*idx)+i))/255.0)-.5;
   sample[INPUTS-1]=1.0;

   // Train the network
   if (mode==0)     // Single layer
    if (sigm==0) cls=train_1layer_net(sample,*(trainingLabels+idx),logistic,w_io);
     else cls=train_1layer_net(sample,*(trainingLabels+idx),tanh,w_io);
   else             // Two-layer
    if (sigm==0) cls=train_2layer_net(sample,*(trainingLabels+idx),logistic,units,w_ih,w_ho);
     else cls=train_2layer_net(sample,*(trainingLabels+idx),tanh,units,w_ih,w_ho);

   if (cls==*(trainingLabels+idx)) correct_class_counter[*(trainingLabels+idx)]++;
   else {
     // Render false negative onto image
     iy=10+((*(trainingLabels+idx))*38);
     ix=10+((fn_counter[*(trainingLabels+idx)])*38);    
     fn_counter[*(trainingLabels+idx)]=(fn_counter[*(trainingLabels+idx)]+1)%10;
     for (int i=0; i<32; i++)
      for (int j=0; j<32; j++)
        false_neg[iy+j][ix+i]=*(trainingDigits+i+(j*32)+((INPUTS-1)*idx));
     // Render false positive onto image
     iy=10+(cls*38);
     ix=10+((fp_counter[cls])*38);    
     fp_counter[cls]=(fp_counter[cls]+1)%10;
     for (int i=0; i<32; i++)
      for (int j=0; j<32; j++)
        false_pos[iy+j][ix+i]=*(trainingDigits+i+(j*32)+((INPUTS-1)*idx));
   }
   total_class_counter[*(trainingLabels+idx)]++;
   count_iters++;  
  }
  
   // Print out the current stats about classification accuracy
   new_avg=0;
   fprintf(stderr,"**** After %d iterations (%d):\n",count_iters,no_improv);
   for (int i=0; i<CLASSES; i++)
   {
    fprintf(stderr,"Class %d, correct classification rate=%f\n",i,(double)correct_class_counter[i]/(double)total_class_counter[i]);
    new_avg+=(double)correct_class_counter[i]/(double)total_class_counter[i];
   }
   new_avg/=(double)CLASSES;
   fprintf(stderr,"Average correct classification rate: %f\n",new_avg);
   memset(&correct_class_counter[0],0,CLASSES*sizeof(int));
   memset(&total_class_counter[0],0,CLASSES*sizeof(int));
   
   // Print out some stats about network weights
   if (mode==0)
   {
    mx=0;
    for (int i=0; i<OUTPUTS; i++)
      for (int j=0; j<INPUTS; j++) 
    if (fabs(w_io[j][i])>mx) mx=fabs(w_io[j][i]);
    fprintf(stderr,"Magnitude of largest network weight: %f\n",mx);
   }
   else
   {
    mx=0;
    for (int i=0; i<OUTPUTS; i++)
      for (int j=0; j<units; j++) 
        if (fabs(w_ho[j][i])>mx) mx=fabs(w_ho[j][i]);
    fprintf(stderr,"Largest hidden to output weight: %f\n",mx);
    mx=0;
    for (int i=0; i<INPUTS; i++)
      for (int j=0; j<units; j++)
        if (fabs(w_ih[j][i])>mx) mx=fabs(w_ih[j][i]);
    fprintf(stderr,"Largest input to hidden weight: %f\n",mx);	 
   }

   // Update the neiwork weights image to display on the frame buffer
   if (mode==0)       // 1-layer - will display the input-to-output weights w_io[][]
   {
    mi=10000;
    mx=-10000;
    for (int i=0; i<10; i++)         
    {
     for (int j=0; j<INPUTS-1; j++)
     {
      *(sample+j)=w_io[j][i];
      if (w_io[j][i]<mi) mi=w_io[j][i];
      if (w_io[j][i]>mx) mx=w_io[j][i];
     }
     for (int j=0; j<INPUTS-1; j++) *(sample+j)=((*(sample+j))-mi)/(mx-mi);
     if (i<5) {iy=10+(3*38); ix=10+(38*2*i);}
     else {iy=10+(5*38); ix=10+(38*2*(i-5));}
     for (int k=0; k<SIDE; k++)
      for (int l=0; l<SIDE; l++)
      {
       network_weights[iy+(2*l)][ix+(2*k)]=(unsigned char)(255*(*(sample+k+(l*SIDE))));           
       network_weights[iy+(2*l)+1][ix+(2*k)]=(unsigned char)(255*(*(sample+k+(l*SIDE))));           
       network_weights[iy+(2*l)][ix+(2*k)+1]=(unsigned char)(255*(*(sample+k+(l*SIDE))));           
       network_weights[iy+(2*l)+1][ix+(2*k)+1]=(unsigned char)(255*(*(sample+k+(l*SIDE))));           
      }
    }
    if (activeFB<=10)
     renderOutputLayerWeights();
    else
     renderFPFN();
   }
   else             // 2-layer will display the hidden layer kernels w_ih[][]
   {      
    // Render current weights onto the OpenGL framebuffer
    memset(&fb[0],0,1024*1024*3*sizeof(unsigned char));
    
    if (activeFB==10)
    {
     for (int i=0; i<units; i++)
     {
        mx=-10000;
        mi=10000;
        for (int j=0; j<INPUTS-1; j++)
        {
          patch[j]=w_ih[j][i];
          if (patch[j]>mx) mx=patch[j];
          if (patch[j]<mi) mi=patch[j];
        }
        for (int j=0; j<INPUTS-1; j++)
          if (mx-mi!=0) patch[j]=(patch[j]-mi)/(mx-mi);
        
        iy=37+((i/25)*38);
        ix=37+((i%25)*38);
                
        for (int k=0; k<SIDE; k++)
          for (int j=0; j<SIDE; j++) 
          {
              *(fb+(((ix+j)+((iy+k)*1024))*3)+0)=(unsigned char)(255*patch[j+(k*SIDE)]);
              *(fb+(((ix+j)+((iy+k)*1024))*3)+1)=(unsigned char)(255*patch[j+(k*SIDE)]);
              *(fb+(((ix+j)+((iy+k)*1024))*3)+2)=(unsigned char)(255*patch[j+(k*SIDE)]);
          }
     }
    }
    else if (activeFB>=0&&activeFB<10)
    {
      if (sigm==0) renderHiddenResponse(activeFB,logistic);
      else renderHiddenResponse(activeFB,tanh);
    }
    else renderFPFN();

    memset(&w2l[0][0],0,INPUTS*10*sizeof(double));
    for (int cc=0; cc<10; cc++)
      for (int i=0; i<units; i++)
        for (int j=0; j<INPUTS; j++)
	      w2l[j][cc]+=w_ih[j][i]*w_ho[i][cc];       

      for (int cc=0; cc<10; cc++)
      {
       mi=10000;
       mx=-10000;
       for (int i=0;i<INPUTS;i++)
       {
        if (w2l[i][cc]<mi) mi=w2l[i][cc];
        if (w2l[i][cc]>mx) mx=w2l[i][cc];
       }
       for (int i=0;i<INPUTS;i++)
	  w2l[i][cc]=(w2l[i][cc]-mi)/(mx-mi);
      }

      for (int cc=0; cc<10; cc++)
      {
       if (cc<5) {iy=10+(3*38); ix=10+(38*2*cc);}
       else {iy=10+(5*38); ix=10+(38*2*(cc-5));}
       for (int k=0; k<SIDE; k++)
        for (int l=0; l<SIDE; l++)
        {
         network_weights[iy+(2*l)][ix+(2*k)]=(unsigned char)(255*w2l[k+(SIDE*l)][cc]);           
         network_weights[iy+(2*l)+1][ix+(2*k)]=(unsigned char)(255*w2l[k+(SIDE*l)][cc]);           
         network_weights[iy+(2*l)][ix+(2*k)+1]=(unsigned char)(255*w2l[k+(SIDE*l)][cc]);           
         network_weights[iy+(2*l)+1][ix+(2*k)+1]=(unsigned char)(255*w2l[k+(SIDE*l)][cc]);           
        }     
      }
    }
    
    // Convergence check	
    if (new_avg<=best_avg) no_improv++;
    else best_avg=new_avg;
    if (no_improv>max_iters)
    {
      doneTrain=1;
      if (mode==2)    // Outputs 2-layer feature maps
      { 
       for (int cc=0; cc<10; cc++)
        if (sigm==0) renderHiddenResponse(cc,logistic);
        else renderHiddenResponse(cc,tanh);
      }
      // Output final weights, false positives, and false negatives
      sprintf(&line[0],"false_negatives_%d_%d_%d.ppm",mode,units,sigm);
      imageConvert(&false_neg[0][0],390,390,line);
      sprintf(&line[0],"false_positives_%d_%d_%d.ppm",mode,units,sigm);
      imageConvert(&false_pos[0][0],390,390,line);
      sprintf(&line[0],"final_network_weights_%d_%d_%d.ppm",mode,units,sigm);
      imageConvert(&network_weights[0][0],390,390,line);
      sprintf(&line[0],"trained_net_weigths_%d_%d_%d.dat",mode,units,sigm);
      f=fopen(line,"w");
      if (f!=NULL) 
       if (mode==0) 
       {
        fwrite(&w_io[0][0],INPUTS*OUTPUTS*sizeof(double),1,f); 
        fclose(f);
       } 
       else 
       {
        fwrite(&w_ih[0][0],INPUTS*MAX_HIDDEN*sizeof(double),1,f); 
        fwrite (&w_ho[0][0],MAX_HIDDEN*OUTPUTS*sizeof(double),1,f); 
        fclose(f);         
       }
      fprintf(stderr,"Training done!\n");  
    }
    
    count_iters=0;
 }
 else if ((mode==1||mode==3)&&!doneTrain)
 {
  memset(&correct_class_counter[0],0,10*sizeof(int));
  memset(&total_class_counter[0],0,10*sizeof(int));
  new_avg=0;
  sprintf(&line[0],"trained_net_weigths_%d_%d_%d.dat",mode-1,units,sigm);
  f=fopen(line,"r");
  if (f!=NULL) 
   if (mode==1) 
   {
    fread(&w_io[0][0],INPUTS*OUTPUTS*sizeof(double),1,f); 
    fclose(f);
   } 
   else {fread(&w_ih[0][0],INPUTS*MAX_HIDDEN*sizeof(double),1,f); fread(&w_ho[0][0],MAX_HIDDEN*OUTPUTS*sizeof(double),1,f); fclose(f);}
  else
   {fprintf(stderr,"Unable to load network weights for mode %d, units %d, sigmoid type %d\n",mode-1,units,sigm); exit(0);}

  fprintf(stderr,"Evaluating performance on test dataset... mode=%d, units=%d, sigmoid=%d\n",mode-1,units,sigm);

  for (int j=0; j<10000; j++)
  {
   for (int i=0; i<INPUTS-1; i++)
    sample[i]=((double)(*(testingDigits+((INPUTS-1)*j)+i))/255.0)-.5;
   sample[INPUTS-1]=1.0;
   if (mode==1)
    if (sigm==0) cls=classify_1layer(sample,*(testingLabels+j),logistic,w_io);
    else cls=classify_1layer(sample,*(testingLabels+j),tanh,w_io);
   else
    if (sigm==0) cls=classify_2layer(sample,*(testingLabels+j),logistic,units,w_ih,w_ho);
    else cls=classify_2layer(sample,*(testingLabels+j),tanh,units,w_ih,w_ho);
   if (cls==*(testingLabels+j)) correct_class_counter[*(testingLabels+j)]++;
   total_class_counter[*(testingLabels+j)]++;
  }
  
  for (int i=0; i<10; i++)
  {
   fprintf(stderr,"Digit %d, correct classification rate=%f\n",i,(double)correct_class_counter[i]/(double)total_class_counter[i]);
   new_avg+=(double)correct_class_counter[i]/(double)total_class_counter[i];
  }
  new_avg/=10.0;
  fprintf(stderr,"Average correct classification rate: %f\n",new_avg);   
  doneTrain=1;
 }
 
 /***** Scene drawing start *********/

 glClearColor(0.01f,0.01f,0.01f,1.0f);
 glDisable(GL_BLEND);
 glDisable(GL_LIGHTING);
 glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

 glMatrixMode(GL_MODELVIEW);
 glLoadIdentity();

 // Set up texture only the first time through this function
 glEnable(GL_TEXTURE_2D);
 if (frame==0)
 {
  glGenTextures( 1, &texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glBindTexture( GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, fb);
  frame++;
 }
 else	// Afterwards just update texture - significantly faster
 {
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,1024,1024,GL_RGB,GL_UNSIGNED_BYTE, fb);
  frame++;
 }
 // Single quad polygon canvas taking up the whole image
 glBegin (GL_QUADS);
 glTexCoord2f (0.0, 0.0);
 glVertex3f (0.0, 0.0, 0.0);
 glTexCoord2f (scale_factor, 0.0);
 glVertex3f (800.0, 0.0, 0.0);
 glTexCoord2f (scale_factor, scale_factor);
 glVertex3f (800.0, 800.0, 0.0);
 glTexCoord2f (0.0, scale_factor);
 glVertex3f (0.0, 800.0, 0.0);
 glEnd ();
    
 glFlush();
 glutSwapBuffers();

 /***** Scene drawing end ***********/

 // Tell glut window to update itself
 glutSetWindow(windowID);
 glutPostRedisplay();
}
