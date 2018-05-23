#include "cluster.h"

void interface(long int *numAssets, long int *numBranches, long int *numSims,
	       long int *noise, double *minProb, 
	       double *Today, double *Growth, double *Covar, double *ansa);

int main(int argc, char **argv) {
  long int numAssets    = 2;
  long int numBranches  = 3;
  long int numSims      = 1000;
  double Today [2] = {100.0, 200.0};
  double Growth[2] = {0.10,  0.05};
  double Covar [4] = {8.0, 1.0, 1.0, 12.0};
  double answer[9]; 
  double minProb = (double)atof(argv[1]);
  long int noise = 2;

  while (1) {
  interface(&numAssets, &numBranches, &numSims, &noise, &minProb, 
	    Today, Growth, Covar, answer);

  for (int i=0; i<3; i++)
    cout << "Branch " << i << " (p = " << answer[3*i+2]
  	 << "):\t" << answer[3*i+0] << "\t" << answer[3*i+1] << endl;
  cout << endl;
  }
}









































