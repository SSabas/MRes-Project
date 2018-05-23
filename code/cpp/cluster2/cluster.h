using namespace std;
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include "myrand.h"
#include "stats.h"
#include "stopwtch.h"

typedef enum{STD, SOBOL, ANTITHETIC} iidtype;
/* Class to assist parsing and remembering options given on command line */
class CmdLine {
 private:
 public:
  char infile[40];
  char outfile[40];
  long int noSims;
  double ratio;
  double minimum;
  int price;
  int method;
  iidtype iidmethod;
  int lognormal;
  int noiseLevel;
  int damp;
  double damp_factor;
  ofstream ofile;

  int noise(int k) {return noiseLevel == k;}
  int Parse(int argc, char **argv);
};


/* scenarioCluster is the central class of program Cluster */
class scenarioCluster {
 private:
 public:
  /* Global problem data, read from input via method ReadData(CmdLine *) */
  static int noAssets;
  static int noStages;
  static int *Brch;
  static long int *minScen;
  static double *Vol;
  static double **Cov;
  static double **Cho;
  static double *Gro;
  static double *Ini;

  double **sc;      // array to hold scenario vectors
  double *centroid; // vector of prices for this node
  double *ret;      // prices / (parent's prices)
  double prob;      // probability: proportional to number of scenarios (n)
  long int n;       // number of scenarios held in sc
  long int maxn;    // size of sc
  int depth;        // == strlen(label): depth in scenario tree
  char label[20];   // label suitable for output to foliage
  scenarioCluster *prev, *next;  // for maintenance of circular DLL
  
  /* This constructor is used directly only for the root node */
  scenarioCluster(long int numba) {
    maxn     = numba;
    sc       = new double*[maxn];
    centroid = new double[noAssets];
    ret      = new double[noAssets];
    n        = depth = 0;
    label[0] = '\0';
    prob     = 0.0;
    next     = 
    prev     = this;
  }

  /* This constructor for creating non-root nodes, takes label, depth
     information from paret (this) */
  scenarioCluster *newChild(long int numba, int childNo) {
    scenarioCluster *nc;
    nc = new scenarioCluster(numba);
    sprintf(nc->label, "%s%d", label, childNo);
    nc->depth = depth+1;
    return nc;
  }

  /* Destructor must clean up allocated memory */
  ~scenarioCluster() {
    if (sc != NULL) delete[] sc;
    delete[] centroid;
    delete[] ret;
  }

  /* p->insertBefore(q) will insert p into q's DLL, so q=p->next */
  scenarioCluster *insertBefore(scenarioCluster *p) {
    next    = p;
    prev    = p->prev;
    p->prev = this;
    prev->next = this;

    return this;
  }

  /* p->spliceBefore(q) will insert DLL containing p into DLL containing q,
     so that p,p->next,...p->prev, q, q->next,...q->prev,p */
  scenarioCluster *spliceBefore(scenarioCluster *p) {
    scenarioCluster *pprev;

    pprev       = p->prev;
    pprev->next = this;
    prev->next  = p;
    p->prev     = prev;
    prev        = pprev;
    return this;
  }

  /* remove single node from DLL */
  scenarioCluster *remove() {
    next->prev = prev;
    prev->next = next;

    next = prev = this;
    return this;
  }

  /* given a vector of double[noAssets], add pointer to end of list sc */
  void addScenario(double *scenario) {
    if (n < maxn)
      sc[n++] = scenario;
    else {
      cerr << "Tried to add scenario to full scenarioCluster" << endl;
      exit(1);
    }
  }

  /* by setting n=0, pointers in sc are ignored */
  void reset() { n=0; }

  int isRoot() { return (depth==0); }

  /* after processing, long array sc is no longer needed */
  void purge() {
    delete[] sc;
    sc = NULL;
  }

  /* print information for this scenario node, in format for foliage */
  void output(CmdLine *C) {
    if (depth>0) {
      C->ofile << label << " " << prob << "\t";
      for (int i=0; i<noAssets; i++) 
	C->ofile << ( C->price ? centroid[i] : ret[i] ) << " ";
      C->ofile << endl;
    }
  }

  /* prototypes for longer methods */
  void readData(CmdLine *C);
  void Header(CmdLine *C, Stats *S);
  void Footer(CmdLine *C);
  void Cholesky();

  void Allocate(long int noSim);
  void Rootify();
  void Update(CmdLine*C);
  void UpdateSTD(CmdLine *C);
  void UpdateSOBOL(CmdLine *C);
  int  Cluster(CmdLine *C);
  void Shift(int discard);
};
  


