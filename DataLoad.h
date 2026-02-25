// Data definitions and function prototypes - We will load the whole CIFAR-10
// dataset, and the program will create training/testing batches as needed

unsigned char labels[60000];
unsigned char images[32*32*60000];

int readCIFAR10(void);
void sampleSplit(int nTrain, int nTest);
