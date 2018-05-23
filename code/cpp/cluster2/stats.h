

class Stats {
 private:
  double* sumx;
  double** sumxy;
  int k,n;
 public:
  Stats(int givenK=1) { 
    int i,j;

    k = givenK;
    sumx  = new double[k];
    sumxy = new double*[k];
    for (i=0; i<k; i++) sumx[i] = 0.0;
    for (i=0; i<k; i++) sumxy[i] = new double[k];
    for (i=0; i<k; i++) 
      for (j=0; j<k; j++) sumxy[i][j] = 0.0;
    n = 0;
  }

  void reset() {
    int i,j;
    for (i=0; i<k; i++) {
      sumx[i] = 0.0;
      for (j=0; j<k; j++) sumxy[i][j] = 0.0;
    }
    n=0;
  }

  int entry(double *x) { 
    int i,j;
    
    for (i=0; i<k; i++) {
      sumx[i] += x[i];
      for (j=0; j<k; j++) sumxy[i][j] += x[i]*x[j];
    }
    n++;
    return n;
  }

  int remove(double *x) {
    int i,j;
    for (i=0; i<k; i++) {
      sumx[i] -= x[i];
      for (j=0; j<k; j++) sumxy[i][j] -= x[i]*x[j];
    }
    n--;
    return n;
  }

  double mean(int i=0) 
    { return sumx[i]/n; }
  double var(int i=0)  
    { return (sumxy[i][i]/(n-1) - sumx[i]*sumx[i]/(n*(n-1))); }
  double stdev(int i=0)
    { return sqrt(var(i)); }
  double covar(int i, int j)
    { return (sumxy[i][j]/(n-1) - sumx[i]*sumx[j]/(n*(n-1))); }
  double correl(int i, int j)
    { return (covar(i,j)/(stdev(i)*stdev(j))); }
  double slope(int x, int y) 
    { return (covar(x,y)/var(x)); }
  double intercept(int x, int y)
    { return (mean(y) - (slope(x,y)*mean(x))); }



};







