/* Sobol generator, courtesy of Benedict Tanyi, FECIT.  Where he got it
   from, I haven't a clue!  -rjs */

#define MAXDIM 160
#define MAXBIT 30


struct SobolData{ 
        //Universal:
        int dimension;
        unsigned long start;
        unsigned long blocksize;
        //Faure & Halton:
        unsigned long n;        /*The sequence index*/
        //Sobol:
        unsigned long in;
        unsigned long ix[MAXDIM+1];
        double fac;
        //Faure:
        int b;           /*base >= dimension*/
        double p[20]; 
};



void sob_initialise();
void sob_set(struct SobolData *counter, int dimension, long start, 
	     long blocksize);
void sobseq(struct SobolData *count, double *x);

#ifndef IN_SOBOL_GENERATOR
extern int sobol_init;
extern unsigned long iv[MAXDIM*MAXBIT+1]; 
#endif

double invnorm(double z);
long uni_ldisc(SobolData *C, long lo, long hi);


