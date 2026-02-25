// Loads the CIFAR-10 data for use with the
// code that trains a NN to perform one-vs-rest 
// classification on images from CIFAR-10
//
// This is just a demo, we don't care about
// generalization, so we'll just use the test
// data which has equal amounts of images from
// each class (apparently this is not true
// of the training batches)
//
// (c) F. Estrada, Jul. 2025

#include "DataLoad.h"

void sampleSplit(int n_train, int n_test)
{
    // Sample CIFAR-10 data into the training and testing sets,
    // The target class is c1, the percentage of images from
    // the training class is pct_c1, and the number of training
    // and testing images are in the corresponding input parameters
    //
    // This leaves the data in the arrays originally meant for
    // MNIST images - ignore the names...
  
    FILE *f2;
    int idx_dataset=0;
    int idx_tmp=0;
    double dice;
    int cnt[10];
    unsigned char *p=&images[0];
        
    // Looks like the dataset is hard, let's see if we can 'simplify it' a bit - testing data will have to go...
    
    
    while (idx_tmp<n_train)
    {
      *(trainingLabels+idx_tmp)=labels[idx_dataset];
      memcpy(trainingDigits+(32*32*idx_tmp),p+(idx_dataset*32*32),32*32*sizeof(unsigned char));
      idx_tmp++;
      idx_dataset++;
    }

    idx_tmp=0;
    while (idx_tmp<n_test)
    {
      *(testingLabels+idx_tmp)=labels[idx_dataset];
      memcpy(testingDigits+(32*32*idx_tmp),p+(idx_dataset*32*32),32*32*sizeof(unsigned char));
      idx_tmp++;
      idx_dataset++;
    }
    
    // Check check
    idx_tmp=0;
    for (int i=0; i<10; i++) cnt[i]=0;
    for (int i=0; i<n_train; i++) cnt[*(trainingLabels+i)]++;
    for (int i=0; i<10; i++)
      fprintf(stderr,"Dataset split for training contains %d (%f of total) images for class %d.\n",cnt[i],(double)cnt[i]/(double)n_train,i);
  
    idx_tmp=0;
    for (int i=0; i<10; i++) cnt[i]=0;
    for (int i=0; i<n_test; i++) cnt[*(testingLabels+i)]++;
    for (int i=0; i<10; i++)
      fprintf(stderr,"Dataset split for testing contains %d (%f of total) images for class %d.\n",cnt[i],(double)cnt[i]/(double)n_test,i);
    
}

int readCIFAR10(void)
{
  // Reads the image data from the CIFAR 10 dataset,
  // taking only the GREEN channel as an approximation
  // to a grayscale image.
  //
  // In the binary file the data is in the following format:
  // 1 unsigned char - label
  // 32*32*3 unsigned char - RGB values
  //
  // Reads the 60000 images, the program will then sample
  // from these to create training/testing sets as needed
  //
  // Returns 1 on success, 0 otherwise

  FILE *f,*f2;
  unsigned char dummy[32*32*3];
  unsigned char *p,*q;
  q=&dummy[0];
  
  memset(&labels[0],0,60000*sizeof(unsigned char));
  memset(&images[0],0,32*32*60000*sizeof(unsigned char));
  
  f=fopen("test_batch.bin","r");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open CIFAR-10 data file. No data!\n");
   return 0;
  }

  for (int i=0; i<10000; i++)
  {
   fread(&labels[i],1*sizeof(unsigned char),1,f);
   fread(&dummy[0],32*32*3*sizeof(unsigned char),1,f);
   p=(&images[0])+(32*32*i);
   memcpy(p,q+1024,1024*sizeof(unsigned char));   
  }
  fclose(f);
  
  f=fopen("data_batch_1.bin","r");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open CIFAR-10 data file. No data!\n");
   return 0;
  }

  for (int i=0; i<10000; i++)
  {
   fread(&labels[i+10000],1*sizeof(unsigned char),1,f);
   fread(&dummy[0],32*32*3*sizeof(unsigned char),1,f);
   p=(&images[0])+(32*32*(10000+i));
   memcpy(p,q+1024,1024*sizeof(unsigned char));   
  }
  fclose(f);
  
  f=fopen("data_batch_2.bin","r");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open CIFAR-10 data file. No data!\n");
   return 0;
  }

  for (int i=0; i<10000; i++)
  {
   fread(&labels[i+20000],1*sizeof(unsigned char),1,f);
   fread(&dummy[0],32*32*3*sizeof(unsigned char),1,f);
   p=(&images[0])+(32*32*(20000+i));
   memcpy(p,q+1024,1024*sizeof(unsigned char));   
  }
  fclose(f);

  f=fopen("data_batch_3.bin","r");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open CIFAR-10 data file. No data!\n");
   return 0;
  }

  for (int i=0; i<10000; i++)
  {
   fread(&labels[i+30000],1*sizeof(unsigned char),1,f);
   fread(&dummy[0],32*32*3*sizeof(unsigned char),1,f);
   p=(&images[0])+(32*32*(30000+i));
   memcpy(p,q+1024,1024*sizeof(unsigned char));   
  }
  fclose(f);

  f=fopen("data_batch_4.bin","r");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open CIFAR-10 data file. No data!\n");
   return 0;
  }

  for (int i=0; i<10000; i++)
  {
   fread(&labels[i+40000],1*sizeof(unsigned char),1,f);
   fread(&dummy[0],32*32*3*sizeof(unsigned char),1,f);
   p=(&images[0])+(32*32*(40000+i));
   memcpy(p,q+1024,1024*sizeof(unsigned char));   
  }
  fclose(f);
  
  f=fopen("data_batch_5.bin","r");
  if (f==NULL)
  {
   fprintf(stderr,"Unable to open CIFAR-10 data file. No data!\n");
   return 0;
  }

  for (int i=0; i<10000; i++)
  {
   fread(&labels[i+50000],1*sizeof(unsigned char),1,f);
   fread(&dummy[0],32*32*3*sizeof(unsigned char),1,f);
   p=(&images[0])+(32*32*(50000+i));
   memcpy(p,q+1024,1024*sizeof(unsigned char));   
  }
  fclose(f);
  
  fprintf(stderr,"Done reading CIFAR-10 data...\n");

  return 1;
}
