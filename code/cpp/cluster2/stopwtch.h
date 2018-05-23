#include <time.h>

class stopwtch{
 private:
 public:
  clock_t start_time;
  clock_t last_time;
  float   off_time;

  /* constructor:  When a new stopwatch is instantiated, check the time */
  stopwtch() {off_time = 0.; last_time = start_time = clock(); }

  void reset() { 
    off_time = 0.; 
    last_time = start_time = clock(); 
  }

  float stop() { 
    float ans = ((clock()-last_time)/((float) CLOCKS_PER_SEC));
    last_time = clock();
    return ans;
  }
                 
  float cumulative() {
    last_time = clock();
    return ((last_time-start_time)/((float) CLOCKS_PER_SEC)) - off_time; 
  }

  float total() {
    last_time = clock();
    return ((last_time-start_time)/((float) CLOCKS_PER_SEC));
  }

  void start() {
    off_time += stop();
  }

};
