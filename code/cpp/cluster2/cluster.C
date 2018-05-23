#include "cluster.h"
#include "sobol.h"

/* distance measure for use by Cluster() */
inline double euclid_sqr(int k, double *x, double *y) {
  double sum=0.0;

  for (int i=0; i<k; i++) sum += (x[i]-y[i])*(x[i]-y[i]);
  return sum;
}

/* ensures that the k randomly chosen points are distinct---otherwise,
   clusters of size 0 will result */
int nonDistinct(int k, long int *x) {
  int i,j;
  for (i=0; i<k; i++)
    for (j=0; j<i; j++)
      if (x[i] == x[j]) return 1;
  return 0;
}


/* matrix scenarioCluster::Cov should already have been inputted in
   method ReadData().  This performs a Cholesky factorization,
   leaving the result in (newly allocated) member Cho                */
void scenarioCluster::Cholesky() {
  int i, j, k;
  double sum;
  
  Cho = new double *[noAssets];
  for (i=0; i<noAssets; i++)  Cho[i] = new double[noAssets];
  
  for(i=0;i<noAssets;i++)
    {
      for(j=0;j<=i;j++)
	{
	  Cho[i][j]=0.;
	  
	  if(j<i){
	    sum = 0.0;
	    for(k=0;k<j;k++)
	      sum += Cho[i][k] * Cho[j][k];
	    
	    Cho[i][j] = (Cov[i][j] - sum)/Cho[j][j];
	  }else{
	    sum = 0.0;
	    for(k=0;k<i;k++)
	      sum += Cho[i][k] * Cho[i][k];
	    if (Cov[i][i] < sum) {
	      cerr << "Negative sqrt in Cholesky!\n\n"; 
	      exit(1);
	    }
	    Cho[i][i] = sqrt(Cov[i][i] - sum);
	  }	
	}
    }
}



/* To be called by the root node: allocate memory for noSim scenarios,
   filling each with the initial prices */
void scenarioCluster::Allocate(long int noSim) {
  long int i;
  double *ary;
  Random R;

  R.clock_seed();
  for (i=0; i<noSim; i++) {
    ary = new double[noAssets];
    addScenario(ary);
  }
}


/* node should have collected all the allocated scenarios while the other
   scenarioCluster's were rounded up and deleted.  Start over with this
   node as root */
void scenarioCluster::Rootify() {
  long int i;
  int j;

  label[0] = '\0';
  depth = 0;
  for (j=0; j<noAssets; j++) centroid[j] = Ini[j];
  for (i=0; i<n; i++) 
    for (j=0; j<noAssets; j++) sc[i][j] = Ini[j];
}

inline double damp(double delta, double p)
{
  return (p * exp(delta/p));
}


/* Just a wrapper to redirect to proper Update routine for parameter
   settings */
void scenarioCluster::Update(CmdLine *C) {
  if (C->iidmethod == SOBOL)
    UpdateSOBOL(C);
  else 
    UpdateSTD(C);
}


/* For each scenario in this node, simulate one time periods' progress */
void scenarioCluster::UpdateSTD(CmdLine *C) {
  long int i, Sims;
  int j;
  double thresh, x;
  Random R;
  static double *noise;
  static int firstTime = 1;

  if (firstTime) {
    noise = new double[noAssets];
    firstTime = 0;
  }

  Sims = (C->iidmethod==ANTITHETIC) ? (n+1)/2 : n;


  for (i=0; i<Sims; i++) if (C->lognormal)
    {
      R.normal(noise, noAssets, Gro, Cho);
      for (j=0; j<noAssets; j++) sc[i][j] *= exp(noise[j]);
      if (C->iidmethod==ANTITHETIC && Sims+i<n)
	for (j=0; j<noAssets; j++) sc[Sims+i][j] *= exp(2*Gro[j]-noise[j]);
    }
  else /* Normal perturbation */
    {
      /* steady growth */
      for (j=0; j<noAssets; j++) {
	sc[i][j] *= (x=exp(Gro[j]));
	if (C->iidmethod==ANTITHETIC && Sims+i<n) sc[Sims+i][j] *= x;
      }
      
      /* plus a random perturbation */
      for (j=0; j<noAssets; j++) noise[j] = 0.0;
      R.normal(noise, noAssets, noise, Cho);
      
      /* protect against negativeness by damping all price drops */
      for (j=0; j<noAssets; j++) {
	if (C->damp) thresh = (1.0-C->damp_factor)*sc[i][j];
	sc[i][j] = (C->damp && -noise[j] > thresh)
	  ? damp(noise[j],sc[i][j]-thresh) + thresh
	  : sc[i][j] + noise[j];
	if (C->iidmethod==ANTITHETIC && Sims+i < n)
	  sc[Sims+i][j] = (C->damp && noise[j] > thresh)
	    ? damp(-noise[j], sc[Sims+i][j])
	    : sc[Sims+i][j] - noise[j];
      }
    }
}

/* Same as UpdateSTD, but use low-discrepancy sequence for uniform
   underlying the normal */
void scenarioCluster::UpdateSOBOL(CmdLine *C) {
  long int i, Sims;
  int j,k;
  double thresh, sum, x;
  static int firstTime = 1;
  static double *noise;
  static struct SobolData counter;

  if (firstTime) {
    firstTime = 0;
    noise = new double[noAssets+1]; // sobol counts from 1
    sob_initialise();
    sob_set(&counter, noAssets, 1, 1);
  }

  Sims = (C->iidmethod==ANTITHETIC) ? (n+1)/2 : n;

  for (i=0; i<Sims; i++) if (C->lognormal)
    {
      sobseq(&counter, noise);           /* iid Uni(0,1) */
      for (j=0; j<noAssets; j++)         /* iid Normal(0,1 */
	noise[j] = invnorm(noise[j+1]);  /* and shift to begin with 1 */

      for (j=0; j<noAssets; j++){ /* incorporate correlation and add to sc */
	for (sum=k=0; k<=j; k++) sum += Cho[j][k]*noise[k];
	sc[i][j] *= exp(Gro[j] + sum);
	if (C->iidmethod==ANTITHETIC && Sims+i<n)
	  sc[Sims+i][j] *= exp(Gro[j] - sum);
      }
    }
  else
    {
      /* steady growth */
      for (j=0; j<noAssets; j++) {
	sc[i][j] *= (x=exp(Gro[j]));
	if (C->iidmethod==ANTITHETIC && Sims+i<n) sc[Sims+i][j] *= x;
      }
      
      /* plus a perturbation */
      sobseq(&counter, noise);           /* iid Uni(0,1)    */
      for (j=0; j<noAssets; j++)         /* iid Normal(0,1) */
	noise[j] = invnorm(noise[j+1]);  /* (and shift to begin with 0) */
      for (j=0; j<noAssets; j++){ /* incorporate correlation and add to sc */
	for (sum=k=0; k<=j; k++) sum += Cho[j][k]*noise[k];
	if (C->damp) thresh = (1.0-C->damp_factor)*sc[i][j];
	sc[i][j] = (C->damp && -sum > thresh)
	  ? damp(sum, sc[i][j]-thresh) + thresh
	  : sc[i][j] + sum;
	if (C->iidmethod==ANTITHETIC && Sims+i<n)
	  sc[Sims+i][j] = (C->damp && sum > thresh)
	    ? damp(-sum, sc[i][j]-thresh) + thresh
	    : sc[i][j] - sum;
	
      }
    }
}


/* For alternative method, where noSims fresh scenarios are branched from
   every node.  

   *this has noSims scenarios, which were supposedly just Cluster()ed.
   *next is the next node in the list to be Cluster()ed

   The list of scenarios pointed to by this node will be initialized to the
   next node's centroid (in preparation for a call to Update**()), and then
   added to the next node's scenario list                                 */
void scenarioCluster::Shift(int discard) {
  /* discard cluster of scenarios that belong to next */
  if (discard) next->reset();  

  /* move scenarios over */
  for (long int i=0; i<n; i++){
    for (int j=0; j<noAssets; j++) sc[i][j] = next->centroid[j];
    next->addScenario(sc[i]);
  }
}




/**********************************************************************
 **** This is the central method of the program.                   ****
 ****                                                              ****
 **** The object 'this' has n scenarios that need to be organized  ****
 **** into clusters, and one child node created for each cluster   ****
 **** Since clustering occurs around randomly chosen scenarios,    ****
 **** it might need to happen more than once.                      ****
 **********************************************************************/

int scenarioCluster::Cluster(CmdLine *C) {
  long int i, bigSize, smlSize, *rands, minSize;
  int j,k,q, minClust, big, sml, tries=0;
  double dist, minDist, rat;
  scenarioCluster **ch;
  static Random R;
  static SobolData Ctr;
  static int firstTime = 1;

  if (firstTime) {
    firstTime = 0;
    if (C->iidmethod == SOBOL) {
      sob_initialise();
      sob_set(&Ctr, 1, 1, 1);
    }
  }
  
  /* A little helpful output */
  q = Brch[depth];            // number of clusters to be formed
  minSize = minScen[depth+1]; // Each cluster must be this big
  if (C->noise(2)) cout << "Node " << label << ":  " << n 
			<< " scenarios into " << q << " clusters "
			<< "(each > " << minSize << "):\n";
  if (isRoot()&&C->noise(1)) cout << "level 0: "; 

  if (C->noise(1)) cout << "|" << flush;


		   
  /* create scenarioCluster structures for the children */
  ch = new scenarioCluster *[q];
  for (j=0; j<q; j++)  // put nodes in job queue
    ch[j] = newChild(maxn, j)->insertBefore(this);
  

  /* Initially, choose first five points as representatives */
  rands = new long int[q];
  for (j=0; j<q; j++) {  // Install representative points into clusters
    rands[j] = j;
    for (k=0; k<noAssets; k++) ch[j]->centroid[k] = sc[j][k];
    ch[j]->reset();
  }

    
  while(1) { // This whole loop performs a random clustering
       // The loop is exited when the clusters meet the size requirement

    if (tries++ > 100) {
	if (C->noise(1)) cout << endl;
	if (C->noiseLevel > 0) 
	  cout << "***Stuck clustering " << n << " scenarios into " << q 
	       << " clusters of size > " << minSize 
	       << ": Restarting***\n\n";
      return(1);
    }
    /* find which cluster (0<j<q) each point (0<i<n) is closest to */
    for (i=0; i<n; i++) {
      minClust=0;
      minDist = euclid_sqr(noAssets, sc[i], ch[0]->centroid);
      for (j=1; j<q; j++)
	if ( (dist=euclid_sqr(noAssets, sc[i], ch[j]->centroid)) < minDist) {
	  minDist  = dist;
	  minClust = j;
	}
      
      /* add the scenario to its appropriate cluster */
      ch[minClust]->addScenario(sc[i]);
    }
    
    /* Now that all the scenarios are in their clusters, find whether the
       cluster sizes are appropriate */
    
    big = sml = 0;
    bigSize = smlSize = ch[0]->n;
    if (C->noise(2)) cout << "\t" << "(" << ch[0]->n;
    for (j=1; j<q; j++) {
      if (C->noise(2)) cout << ", " << ch[j]->n;
      if (ch[j]->n < smlSize) { sml = j; smlSize = ch[j]->n; }
      if (ch[j]->n > bigSize) { big = j; bigSize = ch[j]->n; }
    }
    rat = (1.0*bigSize)/smlSize;
    if (C->noise(2)) cout << ")  Ratio = " << rat << endl;
    if ( (rat < C->ratio) && (C->method || smlSize >= minSize) ) break;

    if (C->noise(1)) cout << "." << flush; // Helpful output to mark failure

    /* For extreme clusters, choose new random representatives */
    do {
      rands[sml] = (C->iidmethod==SOBOL)
                     ? uni_ldisc(&Ctr, 0, n-1)  :  R.uni_ldisc(0,n-1);
      rands[big] = (C->iidmethod==SOBOL)
                     ? uni_ldisc(&Ctr, 0, n-1)  :  R.uni_ldisc(0,n-1);
    } while(nonDistinct(q, rands));

    // Install new representatives into cluster 
    for (k=0; k<noAssets; k++) {
      ch[big]->centroid[k] = sc[rands[big]][k];
      ch[sml]->centroid[k] = sc[rands[sml]][k];
    }
    for (j=0; j<q; j++) ch[j]->reset();  // Empty all clusters
  }

  delete[] rands; // don't need this anymore


  
  /* now that we have an acceptable clustering, set probabilities, choose
     final centroids, and get out of here */
  
  for (j=0; j<q; j++) {// for each cluster, 
    // First, probability
    ch[j]->prob = (double) (1.0*ch[j]->n / n);

    // Then, find "center" (mean for now?)
    for (k=0; k<noAssets; k++) ch[j]->centroid[k] =  0.0;
    for (i=0; i<ch[j]->n; i++) 
      for (k=0; k<noAssets; k++) ch[j]->centroid[k] += ch[j]->sc[i][k];
    for (k=0; k<noAssets; k++) ch[j]->centroid[k] /= ch[j]->n;

    // Then find the simulated point closest to the "center"
    minDist = euclid_sqr(noAssets, ch[j]->centroid, ch[j]->sc[sml=0]);
    for (i=0; i<ch[j]->n; i++) 
      if ((dist=euclid_sqr(noAssets,ch[j]->centroid,ch[j]->sc[i])) < minDist){
	sml = i;
	minDist = dist;
      }

    // And install that point for posterity
    for (k=0; k<noAssets; k++) {
      ch[j]->centroid[k] = ch[j]->sc[sml][k];         /* store the price */
      ch[j]->ret[k] = ch[j]->centroid[k]/centroid[k]; /* and return rate */
    }
  }


  /* A final bit of helpful output */
  if (depth < next->depth && C->noise(1)) {
    if (next->depth < noStages) cout << "\nlevel " << next->depth << ": ";
    else cout << endl;
  }

  delete[] ch;
  return(0);
}
