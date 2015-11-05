#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "logger.h"
#include "types.h"
#include "args.h"
#include "base_mpi.h"

extern "C" void FMM_Init(int images);
extern "C" void FMM_Finalize();
extern "C" void FMM_Partition(int & n, int * index, double * x, double * q, double cycle);
extern "C" void FMM_Coulomb(int n, double * x, double * q, double * p, double * f, double cycle);
extern "C" void Ewald_Coulomb(int n, double * x, double * q, double * p, double * f,
			      int ksize, double alpha, double sigma, double cutoff, double cycle);
extern "C" void Direct_Coulomb(int n, double * x, double * q, double * p, double * f, double cycle);
extern "C" void Direct_Coulomb_TS(int & nti, int & nsi,  double * x, double * x2,
				  double * q, double * q2, double * p);
extern "C" void Test_Sum(int Ni, double * p, double * p2,
			 double * q,
			 double * f, double * f2);
extern "C" void Test_Direct(int Ni, double * p, double * p2,
			 double * q);

using namespace exafmm;

int main(int argc, char ** argv) {

  int Nmax = 0;
  int Ni = 500;
  int stringLength = 20;
  int images = 3;
  int ksize = 11;
  int threads = 16;
  int verbose = 1;
  float cycle = 2 * M_PI;
  float alpha = 10 / cycle;
  float sigma = .25 / M_PI;
  float cutoff = cycle / 2;

  Args * args;
  args = new Args;  
  int mpisize, mpirank;
  BaseMPI * baseMPI;
  baseMPI = new BaseMPI;

   // MPI_Init(&argc, &argv);
  // MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  // MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

  logger::verbose = 1;//args->verbose & (baseMPI->mpirank == 0);  
  args->print(logger::stringLength, P);
  logger::printTitle("FMM Profiling");
  logger::startTimer("Allocation");

  Nmax = args->numBodies;
  int * ibody = new int [Nmax];
  int * icell = new int [Nmax];
  float * x = new float [3*Nmax];
  float * q = new float [Nmax];
  float * p = new float [Nmax];
  float * f = new float [3*Nmax];
  float * p2 = new float [Nmax];
  float * f2 = new float [3*Nmax];

  logger::stopTimer("Allocation");

  logger::startTimer("initV");
  srand48(mpirank);
  double average = 0;
  for (int i=0; i<Ni; i++) {
    x[3*i+0] = drand48() * cycle - cycle / 2;
    x[3*i+1] = drand48() * cycle - cycle / 2;
    x[3*i+2] = drand48() * cycle - cycle / 2;
    p[i] = f[3*i+0] = f[3*i+1] = f[3*i+2] = 0;
    ibody[i] = i + mpirank*Ni;
    icell[i] = ibody[i];
  }
  for (int i=0; i<Ni; i++) {
    q[i] = drand48() - .5;
    average += q[i];
  }
  average /= Ni;
  for (int i=0; i<Ni; i++) {
    q[i] -= average;
  }

  logger::stopTimer("initV");
  logger::printTime("Allocation");
  logger::printTime("initV");
  
  delete[] ibody;
  delete[] icell;
  delete[] x;
  delete[] q;
  delete[] p;
  delete[] f;
  delete[] p2;
  delete[] f2;
  MPI_Finalize();
}
