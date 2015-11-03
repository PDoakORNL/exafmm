#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

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

int main(int argc, char ** argv) {
  const int Nmax = 1000000;
  int Ni = 500;
  int stringLength = 20;
  int images = 6;
  int ksize = 11;
  double cycle = 2 * M_PI;
  double alpha = 10 / cycle;
  double sigma = .25 / M_PI;
  double cutoff = cycle * alpha / 3;
  int * index = new int [Nmax];
  double * x = new double [3*Nmax];
  double * q = new double [Nmax];
  double * p = new double [Nmax];
  double * f = new double [3*Nmax];
  double * p2 = new double [Nmax];
  double * f2 = new double [3*Nmax];

  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

#if 1
  srand48(mpirank);
  double average = 0;
  for (int i=0; i<Ni; i++) {
    x[3*i+0] = drand48() * cycle - cycle / 2;
    x[3*i+1] = drand48() * cycle - cycle / 2;
    x[3*i+2] = drand48() * cycle - cycle / 2;
    p[i] = f[3*i+0] = f[3*i+1] = f[3*i+2] = 0;
  }
  for (int i=0; i<Ni; i++) {
    q[i] = drand48() - .5;
    average += q[i];
  }
  average /= Ni;
  for (int i=0; i<Ni; i++) {
    q[i] -= average;
  }
#else
  std::stringstream name;
  name << "source" << std::setfill('0') << std::setw(4)
       << mpirank << ".dat";
  std::ifstream file(name.str().c_str(),std::ios::in);
  for (int i=0; i<Ni; i++) {
    file >> x[3*i+0];
    file >> x[3*i+1];
    file >> x[3*i+2];
    file >> q[i];
    index[i] = i + mpirank*Ni;
  }
  file.close();
#endif

  FMM_Init(images);
  FMM_Partition(Ni, index, x, q, cycle);
  FMM_Coulomb(Ni, x, q, p, f, cycle);
  for (int i=0; i<Ni; i++) {
    p2[i] = f2[3*i+0] = f2[3*i+1] = f2[3*i+2] = 0;
  }

  Ewald_Coulomb(Ni, x, q, p2, f2, ksize, alpha, sigma, cutoff, cycle);

  Test_Sum(Ni,p, p2,
		  q,
	   f,f2);

  for (int i=0; i<Ni; i++) {
    p2[i] = f2[3*i+0] = f2[3*i+1] = f2[3*i+2] = 0;
  }

  Direct_Coulomb_TS(Ni,
		    Ni,
		    x,
		    x,
		    q,
		    q,
		    p2);

  Test_Direct(Ni,p, p2,
	      q);
  
  delete[] x;
  delete[] q;
  delete[] p;
  delete[] f;
  delete[] p2;
  delete[] f2;
  FMM_Finalize();
  MPI_Finalize();
}
