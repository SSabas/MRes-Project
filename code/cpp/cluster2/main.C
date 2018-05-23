/**************************************************************************/
/*                             C L U S T E R                              */
/*               Randomized multistage scenario generator                 */
/*                    by Reuben J. Settergren, 7/2000                     */
/* Given                                                                  */
/* - Initial prices                                                       */
/* - Growth rates (possibly from an expoential curve fit to recent data)  */
/* - Covariance of deviation from fixed exponential growth                */
/* - A multistage branching structure for a scenario tree                 */
/*                                                                        */
/* Normal (parallel) method:                                              */
/* Program cluster will simulate many multistage scenarios (starting from */
/* the specified Initial prices, and growing according to the growth rate,*/
/* and perturbed by a multinormal distribution with the specified covar-  */
/* iance).  The scenarios will be randomly clustered by their first       */
/* stage prices, and these clusters will each be represented by level one */
/* nodes of the scenario tree.  Each of those scenario clusters will then */
/* be divided into sub-clusters, which will be represented by level two   */
/* scenario nodes, and so on.                                             */
/*                                                                        */
/* Alternate (sequential) method:                                         */
/* Single-stage scenarios will be created and clustered from the root,    */
/* and for the next level, new simulations, initialized to the prices at  */
/* the child nodes, will be performed, also for one stage, etc.           */
/*                                                                        */
/* Finally, the scenario tree will be output in the format required by    */
/* multistage portfolio optimization program foliage.                     */
/**************************************************************************/
#include "cluster.h"


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




int main(int argc, char **argv) {
  CmdLine C;
  scenarioCluster *q;          // queue of nodes to be processed
  scenarioCluster *processed;  // nodes which have been processed
  scenarioCluster *hold;       // dummy
  int OK=0;
  int tries = 0;
  stopwtch timer;
  float time;

  C.Parse(argc, argv);
  hold = new scenarioCluster(10);  // dummy, just to call readData
  hold->readData(&C);              // initialize static members
  delete hold;


  /* START THE CLOCK! */
  timer.start();

  /* Initialize root of scenario tree */
  q = new scenarioCluster(C.noSims);
  q->Allocate(C.noSims);   // allocate memory for scenarios
  q->Rootify();            // initialize all the scenarios to Ini


  while (!OK && tries++ < 10) {
    OK = 1;
    processed = NULL;

    /******* Main Loop ********/
    while(q->depth < q->noStages) {
      
      /* Process node */
      q->Update(&C);  // Perform 1 stage of simulation on q's scenarios
 
      if (q->Cluster(&C)){ // cluster scenarios (children inserted before q)
	OK=0;
	break;
      }
      if (C.method)       // Alternate method: preserve allocated vectors
	q->Shift(1); 
      
      /* move processed node to other list, and proceed to next in queue */
      hold = q;
      q    = q->next;
      hold->remove()->purge(); // remove from queue and release unneeded memory  
      if (hold->isRoot()) processed = hold;  // root starts processed list
      else hold->insertBefore(processed);    // others added to end
    }

    /* if bad exit from clustering, gotta start over */
    if (!OK) {
      /* deallocate scenarioClusters on the processed list */
      if (processed != NULL) {
	while ((hold=processed->next) != (processed->remove())) {
	  delete processed;
	  processed = hold;
	}
	delete hold;
      }

      /* deallocate scenarioClusters on the job queue */
      while (q != (hold=q->next)) {
	q->Shift(0);  // gives all of q's allocated scenarios to hold
	q->remove();
	delete q;
	q = hold;
      }
      
      /* q should be only remaining scenarioCluster: reinitialize as root */
      q->Rootify();
    }
  }


  if (OK) {
    // Now the queue contains only leaves that do not need to be branched
    q->spliceBefore(processed);
    
    
    /* STOP THE CLOCK! */
    time = timer.stop();
    if (C.noiseLevel) cout << time << " seconds\n";

    // cycle through nodes to measure the covariance.
    Stats S(q->noAssets);
    //    for (q = processed->next; q != processed; q = q->next)
    //      S.entry(C.price ? q->centroid : q->ret);
    
    
    
    // Output scenario file, in format for program foliage
    q->Header(&C, &S);    // NAME, ASSETS, BENCHMARKS, COVARIANCES
    for (q = processed->next; q != processed; q = q->next)
      q->output(&C);      // SCENARIOS
    q->Footer(&C);        // BOUNDS, END
    
    exit(0);
  }

  /* else */
  cerr << "I give up!" << endl;
  exit(1);
}









































