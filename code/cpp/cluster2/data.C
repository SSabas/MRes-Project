#include "cluster.h"

#define caseof(x)   if (!strcmp(keyword, (x)))
#define streq(x,y)     (!strcmp((x),(y)))



/* parse command line switches */
int CmdLine::Parse(int ac, char** av) {
  int i=1, OK=0;
  float x;

  infile[0] = outfile[0] = '\0';
  noSims = 100;
  ratio  = 5.0;
  minimum = 3.0;
  damp = price  = method = 0;
  noiseLevel = 1;
  iidmethod = STD;
  lognormal = 1;
  while (i < ac)
    {
      if (streq(av[i], "-f")) 
	{
	  if (i<ac) {
	    strcpy(infile, av[i+1]);
	    i += 2;
	    OK = 1;
	  }
	} 
      else if (streq(av[i], "-o"))
	{
	  if (i<ac) {
	    strcpy(outfile, av[i+1]);
	    ofile.open(outfile);
	    i += 2;
	    OK = 1;
	  }
	}

      else if (streq(av[i], "-n"))
	{
	  if (i<ac) {
	    x = atof(av[i+1]);
	    noSims = (long int) x;
	    i += 2;
	    OK = 1;
	  }
	}
      else if (streq(av[i], "-r"))
	{
	  if (i<ac) {
	    ratio = (double)atof(av[i+1]);
	    i += 2;
	    OK = 1;
	  }
	}
      else if (streq(av[i], "-m"))
	{
	  if (i<ac) {
	    minimum = (double)atof(av[i+1]);
	    i += 2;
	    OK = 1;
	  }
	}
      else if (streq(av[i], "-d"))
	{
	  damp = 1;
	  damp_factor = (double)atof(av[i+1]);
	  i += 2;
	  OK = 1;
	}
      else if (streq(av[i], "-p"))
	{
	  price = 1;
	  i++;
	  OK = 1;
	}
      else if (streq(av[i], "-SEQ"))
	{
	  method = 1;
	  i++;
	  OK = 1;
	}
      else if (streq(av[i], "-SOBOL"))
	{
	  iidmethod = SOBOL;
	  i++;
	  OK = 1;
	}
      else if (streq(av[i], "-ANTITHETIC"))
	{
	  iidmethod = ANTITHETIC;
	  i++;
	  OK = 1;
	}
      else if (streq(av[i], "-NORMAL"))
	{
	  lognormal = 0;
	  i++;
	  OK = 1;
	}
      else if (streq(av[i], "-v"))
	{
	  noiseLevel = 2;
	  i++;
	  OK = 1;
	}
      else if (streq(av[i], "-q"))
	{
	  noiseLevel = 0;
	  i++;
	  OK = 1;
	}
      if (!OK) {
	cerr 
	  << "cluster <OPTIONS>" << endl
	  << "   -f ifile\tinput file" << endl
	  << "   -o ofile\toutput file" << endl
	  << "   -n N    \tnumber of scenarios to generate" 
	  << "           \t\t(can be in exp. notation)" << endl
	  << "   -r ratio\tmaximum ratio big/small cluster size" 
	  << "           \t\t(default = 5.0)" << endl
	  << "   -m ratio\tminimum ratio scenarios/leaves"
	  << "           \t\t(default = 3.0)" << endl
	  << "   -d q    \tdampen tomorrow prices less than q*today\n"
	  << "   -p      \toutput prices instead of rates of return\n" 
	  << "   -SEQ    \tsequential method: generate N scenarios from"
	  << " every node" << endl
	  << "           \t\t(default = parallel method)" << endl
	  << "   -SOBOL  \tuse Sobol low-discrepancy random generator" << endl
	  << "   -ANTITHETIC\tuse antithetic variates" << endl
	  << "   -NORMAL \tinstead of lognormal RV's" << endl
	  << "   -v      \tmore messages to standard output" << endl
	  << "   -q      \tno messages to standard output" << endl
	  << "   -help   \tthis message" << endl << endl;
	exit(1);
      }
      OK=0;
    }

  return (0);
}




//**************************************************************************//
//                    READ DATA FOR SCENARIO GENERATION                     //
//==========================================================================//
void scenarioCluster::readData(CmdLine *C) {
  ifstream fin(C->infile);
  char keyword[20];
  int i,j,OK=1;

  fin >> keyword;
  if (strcmp(keyword, "ASSETS")) exit(1);
  fin >> noAssets;
  if (noAssets<1) exit(1);

  fin >> keyword;
  if (strcmp(keyword, "STAGES")) exit(1);
  fin >> noStages;
  if (noStages < 1) exit(1);

  Brch    = new int[noStages];
  minScen = new long int[noStages+1];
  Vol     = new double[noAssets];
  Gro     = new double[noAssets];
  Ini     = new double[noAssets];
  //  Vol  = new double[noAssets];
  Cov  = new double *[noAssets];
  for (i=0; i<noAssets; i++) 
    Cov[i]  = new double[noAssets];
  
  while(OK) {
    OK = 0;
    fin >> keyword;

         caseof("INITIAL")    { 
      for (i=0; i<noAssets; i++) fin >> Ini[i]; 
      OK = 1;
    } 
    else caseof("GROWTH")     { 
      for (i=0; i<noAssets; i++) fin >> Gro[i]; 
      OK = 1;
    }
    else caseof("COVARIANCE") {
      // read LT covariance matrix
      for (i=0; i<noAssets; i++) 
	for (j=0; j<=i; j++) {
	  fin >> Cov[i][j];
	  Cov[j][i] = Cov[i][j];
	}
      /* now calculate volatility */
      for (i=0; i<noAssets; i++) Vol[i] = sqrt(Cov[i][i]);
      // Converting to correlation
      //      for (i=0; i<noAssets; i++)
      //	for (j=0; j<noAssets; j++) 
      //	  Cor[i][j] /= Vol[i]*Vol[j];
	  
      OK = 1;
    } 
    else caseof("BRANCHING")  {
      for (i=0; i<noStages; i++) fin >> Brch[i];
      OK = 1;
    } 
    else caseof("END") {
      OK = 0;
    }
    else {
      fin.close();
      cerr << "Error in input file: " << C->infile << endl
	   << "Last keyword: " << keyword << endl;
      exit(1);
    }
  }

  /* file was read in ok, so take care of some final details */
  fin.close();

  /* If lognormal, rescale for returns, not prices */
  if (C->lognormal) 
    for (i=0; i<noAssets; i++)
      for (j=0; j<=i; j++)
	Cov[j][i] = Cov[i][j] /= Ini[i]*Ini[j];
  
  /* cholesky factorization of cov now */
  Cholesky();

  /* calculate minimum number of scenarios needed at each level */
  minScen[noStages] = (long int)C->minimum;
  for (j=noStages-1; j>=0; j--) 
    minScen[j] = minScen[j+1] * (Brch[j] + 1); // +1 is for breathing room
  if (!C->method && C->noSims < minScen[0]) {
    cerr << "Error! Need n > " << minScen[0] << " scenarios" << endl;
    exit(1);
  }
}


/* Outputs the initial part of the foliage data file */
void scenarioCluster::Header(CmdLine *C, Stats *S) {
  C->ofile << "NAME " << C->infile << "_cluster\nASSETS " << noAssets
	   << "\nBENCHMARKS ";
  for (int j=0; j<noAssets; j++) C->ofile << 1.0/noAssets << " ";
  C->ofile << "\nCOVARIANCES" << endl;

  for (int i=0; i<noAssets; i++)
    for (int j=0; j<=i; j++) 
      C->ofile << i << " " << j << " " 
	       << ((C->lognormal) ? Cov[i][j] : Cov[i][j]/(Ini[i]*Ini[j])) 
	       << endl;
  C->ofile << "SCENARIOS" << endl;
}

/* Outputs the final part of the foliage data file */
void scenarioCluster::Footer(CmdLine *C) {
  C->ofile << "BOUNDS\nLWPS R  ";
  for (int i=0; i<noAssets; i++) C->ofile << "  0.0";
  C->ofile << "\nUWPS R  "; 
  for (int i=0; i<noAssets; i++) C->ofile << "  0.5";//3.0/noAssets << " ";
  C->ofile << "\nLXAS R  ";
  for (int i=0; i<noAssets; i++) C->ofile << " -0.5";
  C->ofile << "\nUXAS R  ";
  for (int i=0; i<noAssets; i++) C->ofile << "  0.5";
  C->ofile << "\nEND\n";

  C->ofile.close();
}

