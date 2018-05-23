#include "cluster.h"
#include "sobol.h"
#include "myrand.h"

/**** Initialization of static members for classes ****/
int       scenarioCluster::noAssets = 0;
int       scenarioCluster::noStages = 0;
int      *scenarioCluster::Brch= NULL;
long int *scenarioCluster::minScen = NULL;
double   *scenarioCluster::Vol = NULL;
double  **scenarioCluster::Cov = NULL;
double  **scenarioCluster::Cho = NULL;
double   *scenarioCluster::Gro = NULL;
double   *scenarioCluster::Ini = NULL;
long int Random::idum  = 1;
long int Random::idum2 = 123456789;
long int Random::iy = 0;
long int Random::iv[32] =
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
double Random::aplus = 0.;
double Random::amnus = 0.;


/***************************************************************************
 Standard C interface to clustering algorithm.

INPUT: numAssets, numBranches, numSims, noise, minProb: are pointers to scalars
       noise can be 0 (none), 1 (a little), or 2 (a lot)
       minProb is the smallest tolerated probability of any branch
       Today points to first element of numAssets-vector of today's prices
       Growth points to first element of numAssets-vector of growth rates
       Covar is pointer to first element of (numAssets X numAssets)
             one-dimension array containing covariance matrix
       (since covariance matrices are symmetric, row- or column-major
       shouldn't matter)
OUTPUT: ansa points to numBranches*(numAssets+1) memory reserved for 
       writing the final answer, in the form
       Branch 1(Assets 1...numAssets, prob), 
       Branch 2(Assets 1...numAssets, prob), etc.

The interface routine 
* initializes the appropriate C++ objects
* uses Today/Growth/Covar data to simulate numSims growth events
* calls scenarioCluster::Cluster(cmdLine &C) to use clustering to reduce
  the numSims simulations to numBranches events
* copies the answer into ansa, and returns
***************************************************************************/
void interface(long int *numAssets, long int *numBranches, long int *numSims,
	       long int *noise, double *minProb, 
	       double *Today, double *Growth, double *Covar, double *ansa) {
  int i,j;
  scenarioCluster *q;

  // An object of class cmdLine C is needed to pass to scenarioCluster::Cluster
  CmdLine C;
  C.noSims     = *numSims;
  C.ratio      = 5.0;
  C.minimum    = (*numSims) * (*minProb);
  C.damp       = C.method = 0;
  C.price      = 1;
  C.noiseLevel = *noise;   // 0=silent, 1=|..., 2=description
  C.iidmethod  = SOBOL;
  C.lognormal  = 1;

  // Copy the covariances into 2D array
  double **covHold;
  covHold = new (double *)[*numAssets];
  for (i=0; i<*numAssets; i++) covHold[i] = new double[*numAssets];
  for (i=0; i<*numAssets; i++)
    for (j=0; j<*numAssets; j++) 
      covHold[i][j] =  Covar[i*(*numAssets) + j] / (Today[i]*Today[j]);
  // rescale to return covariances instead of price

  // Initialization of static members of class scenarioCluster
  int brchHold = *numBranches;
  long int minScenHold[2];
  minScenHold[1] = (long int) C.minimum;
  minScenHold[0] = *numSims;

  scenarioCluster::noAssets = *numAssets;
  scenarioCluster::noStages = 1;
  scenarioCluster::Brch     = &brchHold;
  scenarioCluster::minScen  = minScenHold;
  scenarioCluster::Cov      = covHold;
  scenarioCluster::Gro      = Growth;
  scenarioCluster::Ini      = Today;
  

  /* Ready to begin the actual simulation & clustering */
  scenarioCluster root(*numSims);  // allocate today's node
  root.Cholesky();                 // factorize covariance matrix
           // For debugging: print out Covariance and Cholesky matrices
           //  for (i=0; i<*numAssets; i++, cout << endl)
           //    for (j=0; j<*numAssets; j++) cout << root.Cov[i][j] << " ";
           //  for (i=0; i<*numAssets; i++, cout << endl)
           //    for (j=0; j<*numAssets; j++) cout << root.Cho[i][j] << " ";
  root.Allocate(*numSims); // allocate arrays to hold numSims price vectors
  root.Rootify();          // initialize arrays to Today prices
  root.UpdateSOBOL(&C);    // simulate to grow each scenario
           // For debugging: print out the simulated scenarios
           //  for (i=0; i<*numSims; i++, cout << endl)
           //    for (j=0; j<*numAssets; j++) cout << root.sc[i][j] << " ";
  if (root.Cluster(&C))    // perform the clustering
    {                      // if cluster returns nonzero value, it failed
      cerr << "Cluster failed: probability restriction too stringent?\n";
      exit(1);
    }


  // Copy the resulting prices/probabilities into ansa
  int pos=0;
  for (q = root.next; q != &root; q=q->next) {
           //    cout << "copying node " << q->label << endl; // debugging
        for (i=0; i<*numAssets; i++) {
          ansa[pos++] = q->centroid[i];
           //    cout << q->centroid[i] << endl;
    }
    ansa[pos++] = q->prob;
  }


  // clean up memory
  for (i=0; i<*numAssets; i++) {
    delete[] covHold[i];
    delete[] root.Cho[i];
  }
  delete covHold;
  delete root.Cho;
  while (&root != root.next) delete (root.next->remove());
}
