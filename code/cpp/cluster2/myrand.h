#ifndef MYRAND
#define MYRAND

#include <time.h>

/* copied from Numerical Recipes in C, 2d ed., p. 282 */

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)


class Random {
private:
  static long int idum;
  static long int idum2;
  static long int iy;   
  static long int iv[NTAB];
  static double aplus;
  static double amnus;


public:
  Random() {
    aplus = sqrt(2.0/exp(1.0));
    amnus = -aplus;
  }

  void seed(long int i) { idum = i; }
  long int clock_seed() {idum = (long int)time(NULL); return idum;}
  
  double ran2()
    {
      int j;
      long k;
      double temp;
      
      if (idum <=0)
	{
	  if (-(idum) < 1) idum=1;
	  else idum = -(idum);
	  idum2=(idum);
	  for (j=NTAB+7;j>=0;j--)
	    {
	      k=(idum)/IQ1;
	      idum=IA1*(idum-k*IQ1)-k*IR1;
	      if (idum < 0) idum += IM1;
	      if (j < NTAB) iv[j] = idum;
	    }
	  iy=iv[0];
	}
      k=(idum)/IQ1;
      idum=IA1*(idum-k*IQ1)-k*IR1;
      if (idum < 0) idum += IM1;
      k=idum2/IQ2;
      idum2=IA2*(idum2-k*IQ2)-k*IR2;
      if (idum2 < 0) idum2 += IM2;
      j=iy/NDIV;
      iy=iv[j]-idum2;
      iv[j] = idum;
      if (iy<1) iy += IMM1;
      if ((temp=AM*iy) > RNMX) return RNMX;
      else return temp;
    }
  
  double uni_cont(double x=0.0, double y=1.0) {
    static double u;
    u = ran2();
    return (x + u*(y-x));
  }

  int uni_disc(int m, int n)
  {
    return (m + (int) ((n-m+1)*ran2()));
  }

  long int uni_ldisc(long int m, long int n)
  {
    return (m + (long int) ((n-m+1)*ran2()));
  }

  /* Ratio-of-uniforms method with two-sided squeezing, from
     _Non-Uniform_Random_Variate_Generation_, Luc Devroye, pp. 194--- */
  double normal()
    {
      static double u;
      static double v;
      static double x;

      while(1) {
	u = ran2();
	v = uni_cont(aplus, amnus);        // a = +/- sqrt(2/e)
	x = v/u;
	if ( x*x <= 6.0 - 8*u + 2*u*u ||   // quick acceptance
	    (x*x < 2.0/u - 2.0*u &&        // (no) quick rejection
	     x*x <= -4*log(u)))            // full acceptance
	  return (x);
      }
    }

  /* An k-vector of iid N(0,1) variables */
  void normal(double *x, int k) {
    for (int i=0; i<k; i++) x[i] = normal();
  }

  /* An k-vector of iid N(mu,sigma2) variables */
  void normal(double *x, int k, double *mu, double *sigma) {
    for (int i=0; i<k; i++) x[i] = mu[i] + sigma[i]*normal();
  }
    

  /* A k-vector of N(mu,mat*mat') variables, where mat is lower
     triangular, and mat*mat' is a (psd) covariance matrix */ 
  void normal(double *x, int k, double *mu, double **mat) 
    {
      static double temp[100];
      int i,j;
      double sum;
      
      normal(temp, k);
      for (i=0; i<k; i++) {
	sum=0.0;
	for (j=0; j<=i; j++)
	  sum += mat[i][j]*temp[j];
	x[i] = mu[i] + sum;
      }
    }
  
};






#endif
