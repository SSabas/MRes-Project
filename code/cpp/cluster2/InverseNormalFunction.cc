//**************************************************************************//
//-------------------------------------------------------------//
//                                                             //
//   Scalar Inverse Normal Tranformation Function              //
//                                                             //
//-------------------------------------------------------------//
//
/***************************************************************************

Inverse cumulative normal funtion using Moro's algorithm.

Usage: invnorm(arg) returns the inverse of arg (all of type double).

This is faster than the standard Box-Muller algorithm. This also does
not reorder the sequence (by discarding points). Moro states the maximum
error is 3 * 10^-9 . This can be used in the range 10^-10 < arg < 1 - 10^-10.

****************************************************************************/

double invnorm(double z)
{
  int n;
  double y;
  double asum = 0.0;
  double bsum = 0.0;
  double result;
  const static double a[] = 
            {2.50662823844, -18.61500062529, 41.39119773534, -25.44106049637};
  const static double b[] = 
        {1.0, -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833};
  const static double c[] = 
               {7.7108870705487895, 2.7772013533685169, 0.3614964129261002,
		0.0373418233434554, 0.0028297143036967, 0.0001625716917922,
		0.0000080173304740, 0.0000003840919865, 0.0000000129707170};
  const static double k[] = { 0.4179886424926431, 4.2454686881376569 };
  double d[10];

  /* Check input is between 0 and 1. */
  if( (z<=0) || (z>=1) ) {
    fprintf(stderr, "invnorm: Error. Cannot calculate inverse cumulative"
	    " normal of x with x <= 0 or x >= 1\n");
    exit(1);
  }

  y = z - 0.5;

  /* central region approximation */
  if( (0.08<=z) && (z<=0.92) ) {

    for(n=0;n<4;n++)
      asum += a[n] * pow(y, 2*n);

    for(n=0;n<5;n++)
      bsum += b[n] * pow(y, 2*n);

    result = y * asum / bsum;

  } else { 

    /* Tails approximation */

    y = k[0] * (2 * log( - log(0.5 - fabs(y)) ) - k[1]);


    d[9] = 0.0;
    d[8] = c[8];
    for(n=7;n>0;n--)
      d[n] = 2*y*d[n+1] - d[n+2] + c[n];

    result = y*d[1] - d[2] + 0.5 * c[0];

    if(z<0.08) result = - result;

  }

  return result;
}
 
