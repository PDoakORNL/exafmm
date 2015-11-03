#include "base_mpi.h"
#include "args.h"
#include "bound_box.h"
#include "build_tree.h"
#include "ewald.h"
#include "logger.h"
#include "partition.h"
#include "traversal.h"
#include "tree_mpi.h"
#include "up_down_pass.h"
#if MASS
#error Turn off MASS for this wrapper
#endif

using namespace exafmm;

Args * args;
BaseMPI * baseMPI;
BoundBox * boundBox;
BuildTree * localTree, * globalTree;
Partition * partition;
Traversal * traversal;
TreeMPI * treeMPI;
UpDownPass * upDownPass;
Bodies buffer;
Bounds localBounds;
Bounds globalBounds;

extern "C" void Test_Init(int images) {
  printf("in images\n");
}

extern "C" void Test_Init2(int images) {
  printf("in images %d\n ", images);
}

extern "C" int get_mpirank(void) {
  return baseMPI->mpirank;
}

extern "C" void mpi_finalize(void) {
  MPI_Finalize();
}

extern "C" void FMM_Init(int images) {
  const int ncrit = 32;
  const int nspawn = 1000;
  const real_t eps2 = 0.0;
  const real_t theta = 0.4;
  const bool useRmax = true;
  const bool useRopt = true;
  args = new Args;
  baseMPI = new BaseMPI;
  boundBox = new BoundBox(nspawn);
  localTree = new BuildTree(ncrit, nspawn);
  globalTree = new BuildTree(1, nspawn);
  partition = new Partition(baseMPI->mpirank, baseMPI->mpisize);
  traversal = new Traversal(nspawn, images);
  treeMPI = new TreeMPI(baseMPI->mpirank, baseMPI->mpisize, images);
  upDownPass = new UpDownPass(theta, useRmax, useRopt);

  args->theta = theta;
  args->ncrit = ncrit;
  args->nspawn = nspawn;
  args->images = images;
  args->mutual = 0;
  args->verbose = 1;
  args->distribution = "external";
  args->verbose &= baseMPI->mpirank == 0;
  logger::verbose = args->verbose;
  logger::printTitle("Initial Parameters");
  args->print(logger::stringLength, P);
}

extern "C" void FMM_Finalize() {
  delete args;
  delete baseMPI;
  delete boundBox;
  delete localTree;
  delete globalTree;
  delete partition;
  delete traversal;
  delete treeMPI;
  delete upDownPass;
}

extern "C" void FMM_Partition(int & n, int * index, double * x, double * q, double cycle) {
  logger::printTitle("Partition Profiling");
  const int shift = 29;
  const int mask = ~(0x7U << shift);
  Bodies bodies(n);
  logger::printTitle("Partition Profiling --Bodies");
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    int i = B-bodies.begin();
    B->X[0] = x[3*i+0];
    B->X[1] = x[3*i+1];
    B->X[2] = x[3*i+2];
    B->SRC = q[i];
    int iwrap = wrap(B->X, cycle);
    B->IBODY = index[i] | (iwrap << shift);
  }
  localBounds = boundBox->getBounds(bodies);
  globalBounds = baseMPI->allreduceBounds(localBounds);
  localBounds = partition->octsection(bodies,globalBounds);
  bodies = treeMPI->commBodies(bodies);
  Cells cells = localTree->buildTree(bodies, buffer, localBounds);
  upDownPass->upwardPass(cells);

  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    int i = B-bodies.begin();
    index[i] = B->IBODY & mask;
    int iwrap = unsigned(B->IBODY) >> shift;
    unwrap(B->X, cycle, iwrap);
    x[3*i+0] = B->X[0];
    x[3*i+1] = B->X[1];
    x[3*i+2] = B->X[2];
    q[i]     = B->SRC;
  }
  n = bodies.size();
}

extern "C" void FMM_Partition_NonP(int & n, int * index, double * x, double * q) {
  logger::printTitle("Partition Profiling");
  Bodies bodies(n);
  //std::cout << "n:" << n << "   x[0]:" << x[0] << "   q[0]" << q[0] << std::endl;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    int i = B-bodies.begin();
    B->X[0] = x[3*i+0];
    B->X[1] = x[3*i+1];
    B->X[2] = x[3*i+2];
    B->SRC = q[i];
    B->IBODY = index[i];
  }
  localBounds = boundBox->getBounds(bodies);
  //std::cout << "localBounds Xmin[0]" << localBounds.Xmin[0] << std::endl;
  globalBounds = baseMPI->allreduceBounds(localBounds);
  //std::cout << "globalBounds Xmin[0]" << globalBounds.Xmin[0] << std::endl;
  localBounds = partition->octsection(bodies,globalBounds);
  bodies = treeMPI->commBodies(bodies);
  Cells cells = localTree->buildTree(bodies, buffer, localBounds);
  upDownPass->upwardPass(cells);

  n = bodies.size();
}

extern "C" void FMM_Coulomb(int n, double * x, double * q, double * p, double * f, double cycle) {
  args->numBodies = n;
  logger::printTitle("FMM Parameters");
  args->print(logger::stringLength, P);
  logger::printTitle("FMM Profiling");
  logger::startTimer("Total FMM");
  logger::startPAPI();
  Bodies bodies(n);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    int i = B-bodies.begin();
    B->X[0] = x[3*i+0];
    B->X[1] = x[3*i+1];
    B->X[2] = x[3*i+2];
    wrap(B->X, cycle);
    B->SRC = q[i];
    B->TRG[0] = p[i];
    B->TRG[1] = f[3*i+0];
    B->TRG[2] = f[3*i+1];
    B->TRG[3] = f[3*i+2];
    B->IBODY = i;
  }
  Cells cells = localTree->buildTree(bodies, buffer, localBounds);
  upDownPass->upwardPass(cells);
  treeMPI->allgatherBounds(localBounds);
  treeMPI->setLET(cells, cycle);
  treeMPI->commBodies();
  treeMPI->commCells();
  traversal->initWeight(cells);
  traversal->dualTreeTraversal(cells, cells, cycle, args->mutual);
  Cells jcells;
  if (args->graft) {
    treeMPI->linkLET();
    Bodies gbodies = treeMPI->root2body();
    jcells = globalTree->buildTree(gbodies, buffer, globalBounds);
    treeMPI->attachRoot(jcells);
    traversal->dualTreeTraversal(cells, jcells, cycle, false);
  } else {
    for (int irank=0; irank<baseMPI->mpisize; irank++) {
      treeMPI->getLET(jcells, (baseMPI->mpirank+irank)%baseMPI->mpisize);
      traversal->dualTreeTraversal(cells, jcells, cycle, false);
    }
  }
  upDownPass->downwardPass(cells);
  vec3 localDipole = upDownPass->getDipole(bodies,0);
  vec3 globalDipole = baseMPI->allreduceVec3(localDipole);
  int numBodies = baseMPI->allreduceInt(bodies.size());
  upDownPass->dipoleCorrection(bodies, globalDipole, numBodies, cycle);
  logger::stopPAPI();
  logger::stopTimer("Total FMM");
  logger::printTitle("Total runtime");
  logger::printTime("Total FMM");
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    int i = B->IBODY;
    p[i]     = B->TRG[0];
    f[3*i+0] = B->TRG[1];
    f[3*i+1] = B->TRG[2];
    f[3*i+2] = B->TRG[3];
  }
}

extern "C" void Ewald_Coulomb(int n, double * x, double * q, double * p, double * f,
			      int ksize, double alpha, double sigma, double cutoff, double cycle) {
  Ewald * ewald = new Ewald(ksize, alpha, sigma, cutoff, cycle);
  args->numBodies = n;
  logger::printTitle("Ewald Parameters");
  args->print(logger::stringLength, P);
  ewald->print(logger::stringLength);
  logger::printTitle("Ewald Profiling");
  logger::startTimer("Total Ewald");
  logger::startPAPI();
  Bodies bodies(n);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    int i = B-bodies.begin();
    B->X[0] = x[3*i+0];
    B->X[1] = x[3*i+1];
    B->X[2] = x[3*i+2];
    wrap(B->X, cycle);
    B->SRC = q[i];
    B->TRG[0] = p[i];
    B->TRG[1] = f[3*i+0];
    B->TRG[2] = f[3*i+1];
    B->TRG[3] = f[3*i+2];
    B->IBODY = i;
  }
  Cells cells = localTree->buildTree(bodies, buffer, localBounds);
  Bodies jbodies = bodies;
  for (int i=0; i<baseMPI->mpisize; i++) {
    if (args->verbose) std::cout << "Ewald loop           : " << i+1 << "/" << baseMPI->mpisize << std::endl;
    treeMPI->shiftBodies(jbodies);
    localBounds = boundBox->getBounds(jbodies);
    Cells jcells = localTree->buildTree(jbodies, buffer, localBounds);
    ewald->wavePart(bodies, jbodies);
    ewald->realPart(cells, jcells);
  }
  ewald->selfTerm(bodies);
  logger::stopPAPI();
  logger::stopTimer("Total Ewald");
  logger::printTitle("Total runtime");
  logger::printTime("Total Ewald");
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    int i = B->IBODY;
    p[i]     = B->TRG[0];
    f[3*i+0] = B->TRG[1];
    f[3*i+1] = B->TRG[2];
    f[3*i+2] = B->TRG[3];
  }
  delete ewald;
}

void MPI_Shift(double * var, int &nold, int mpisize, int mpirank) {
  const int isend = (mpirank + 1          ) % mpisize;
  const int irecv = (mpirank - 1 + mpisize) % mpisize;
  //std::cout << "mpirank: " << mpirank << "   isend:" << isend << "   irecv:" << irecv << std::endl;
  int nnew;
  MPI_Request sreq, rreq;
  MPI_Isend(&nold, 1, MPI_INT, irecv, 0, MPI_COMM_WORLD, &sreq);
  MPI_Irecv(&nnew, 1, MPI_INT, isend, 0, MPI_COMM_WORLD, &rreq);
  MPI_Wait(&sreq, MPI_STATUS_IGNORE);
  MPI_Wait(&rreq, MPI_STATUS_IGNORE);
  //std::cout << "buff size is " << nnew << std::endl;
  double * buf = new double [nnew];
  MPI_Isend(var, nold, MPI_DOUBLE, irecv, 1, MPI_COMM_WORLD, &sreq);
  MPI_Irecv(buf, nnew, MPI_DOUBLE, isend, 1, MPI_COMM_WORLD, &rreq);
  MPI_Wait(&sreq, MPI_STATUS_IGNORE);
  MPI_Wait(&rreq, MPI_STATUS_IGNORE);
  for (int i=0; i<nnew; i++) {
    var[i] = buf[i];
  }
  nold = nnew;
  delete[] buf;
}

extern "C" void Direct_Coulomb_TS(int & Nti,
				  int & Nsi,
				  double * xtarg,
				  double * xsourcei,
				  double * qtarg,
				  double * qsourcei,
				  double * p) {
  //  std::cout << "damn" <<std::endl;
  int Nt = Nti;
  int Ns = Nsi;
  int Ns3 = 3 * Ns;

  if (baseMPI->mpirank == 0) std::cout << "--- MPI direct sum presource copy" << std::endl;
  
  double * xsource = new double [3*Nsi];
  double * qsource = new double [Nsi];
  for (int i=0; i<Nsi; i++) {
    xsource[3*i+0] = xsourcei[3*i+0];
    xsource[3*i+1] = xsourcei[3*i+1];
    xsource[3*i+2] = xsourcei[3*i+2];
    qsource[i] = qsourcei[i];
  }

  if (baseMPI->mpirank == 0) std::cout << "--- MPI direct sum ---------------" << std::endl;
  for (int irank=0; irank<baseMPI->mpisize; irank++) {
    if (baseMPI->mpirank == 0) std::cout << "Direct loop          : " << irank+1 << "/" << baseMPI->mpisize << std::endl;
    MPI_Shift(xsource, Ns3, baseMPI->mpisize, baseMPI->mpirank); //Changes xsource and Ns3
    MPI_Shift(qsource, Ns,  baseMPI->mpisize, baseMPI->mpirank); //Changes qsource and Ns
    for (int i=0; i<Nt; i++) {
      double pp = 0;
      for (int j=0; j<Ns; j++) {
	double dx = xtarg[3*i+0] - xsource[3*j+0];
	double dy = xtarg[3*i+1] - xsource[3*j+1];
	double dz = xtarg[3*i+2] - xsource[3*j+2];
        double R2 = dx * dx + dy * dy + dz * dz;
        double invR = 1 / std::sqrt(R2);
        if (R2 == 0) invR = 0;
        // double invR3 = qsource[j] * invR * invR * invR;
        pp += qsource[j] * invR;
        // fx += dx * invR3;
        // fy += dy * invR3;
        // fz += dz * invR3;  
      }
      p[i] += pp * qtarg[i];
      // f[3*i+0] -= fx; // what the hell are these?
      // f[3*i+1] -= fy;
      // f[3*i+2] -= fz;
    }
  }
  if (baseMPI->mpirank == 0) std::cout << "Direct loop done" <<std::endl;
  // double localDipole[3] = {0, 0, 0};
  // for (int i=0; i<Ni; i++) {
  //   for (int d=0; d<3; d++) localDipole[d] += x[3*i+d] * q[i];
  // }
  // int N;
  // MPI_Allreduce(&Ni, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  // double globalDipole[3];
  // MPI_Allreduce(localDipole, globalDipole, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // double norm = 0;
  // for (int d=0; d<3; d++) {
  //   norm += globalDipole[d] * globalDipole[d];
  // }
  // double coef = 4 * M_PI / (3 * cycle * cycle * cycle);
  // for (int i=0; i<Ni; i++) {
  //   p[i] -= coef * norm / N / q[i];
  //   f[3*i+0] -= coef * globalDipole[0];
  //   f[3*i+1] -= coef * globalDipole[1];
  //   f[3*i+2] -= coef * globalDipole[2];
  // }
  delete[] xsource;
  delete[] qsource;
}

extern "C" void Direct_Coulomb_LJ_TS(int & Nti,
				     int & Nsi,
				     double * xtarg,
				     double * xsourcei,
				     double * qtarg,
				     double * qsourcei,
				     double * ljtarg,
				     double * ljsourcei,
				     double * pes,
				     double * plj,
				     double eps) {
  int Nt = Nti;
  int Ns = Nsi;
  int Ns3 = 3 * Ns;

  if (baseMPI->mpirank == 0) std::cout << "--- MPI direct sum presource copy" << std::endl;
  
  double * xsource = new double [3*Nsi];
  double * qsource = new double [Nsi];
  double * ljsource = new double [Nsi];
  for (int i=0; i<Nsi; i++) {
    xsource[3*i+0] = xsourcei[3*i+0];
    xsource[3*i+1] = xsourcei[3*i+1];
    xsource[3*i+2] = xsourcei[3*i+2];
    qsource[i] = qsourcei[i];
    ljsource[i] = ljsourcei[i];
  }

  if (baseMPI->mpirank == 0) std::cout << "--- MPI direct LJES sum ---------------" << std::endl;
  for (int irank=0; irank<baseMPI->mpisize; irank++) {
    if (baseMPI->mpirank == 0) std::cout << "Direct loop          : " << irank+1 << "/" << baseMPI->mpisize << std::endl;
    MPI_Shift(xsource, Ns3, baseMPI->mpisize, baseMPI->mpirank); //Changes xsource and Ns3
    MPI_Shift(qsource, Ns,  baseMPI->mpisize, baseMPI->mpirank); //Changes qsource and Ns
    for (int i=0; i<Nt; i++) {
      double pp = 0;
      double lje = 0;
      for (int j=0; j<Ns; j++) {
	double dx = xtarg[3*i+0] - xsource[3*j+0];
	double dy = xtarg[3*i+1] - xsource[3*j+1];
	double dz = xtarg[3*i+2] - xsource[3*j+2];
        double R2 = dx * dx + dy * dy + dz * dz;
	double R = std::sqrt(dx * dx + dy * dy + dz * dz);
        double invR = 1 / R; //std::sqrt(R2);
	if (R2 == 0) invR = 0; //issue here of taking sqrt of 0
	double sigma = ljtarg[i] + ljsource[j];
	if(invR < 1/(2.5 * sigma)) {
	  lje += 0;
	}
	else {
	  double sod = sigma/R;
          double att = 2 * sod * sod * sod * sod * sod * sod;
	  double sodc = sigma/(2.5*sigma);
          double attc = 2 * sodc * sodc * sodc * sodc * sodc * sodc;
          double rep = att * att;
          double repc = attc * attc;
          lje +=  eps  * (rep - att - (repc- attc)); //ideally eps is pairwise too
	}
	pp += qsource[j] * invR;
      }

        // fx += dx * invR3;
        // fy += dy * invR3;
        // fz += dz * invR3;  
      pes[i] += pp * qtarg[i];
      plj[i] += lje;
    }

      // f[3*i+0] -= fx; // what the hell are these?
      // f[3*i+1] -= fy;
      // f[3*i+2] -= fz;
  }
  if (baseMPI->mpirank == 0) std::cout << "Direct loop done" <<std::endl;
  delete[] xsource;
  delete[] qsource;
  delete[] ljsource;
}

extern "C" void Direct_Coulomb(int Ni, double * x, double * q, double * p, double * f, double cycle) {
  const int Nmax = 1000000;
  int images = args->images;
  int prange = 0;
  for (int i=0; i<images; i++) {
    prange += int(std::pow(3.,i));
  }
  double * x2 = new double [3*Nmax];
  double * q2 = new double [Nmax];
  for (int i=0; i<Ni; i++) {
    x2[3*i+0] = x[3*i+0];
    x2[3*i+1] = x[3*i+1];
    x2[3*i+2] = x[3*i+2];
    q2[i] = q[i];
  }
  double Xperiodic[3];
  int Nj = Ni, Nj3 = 3 * Ni;
  if (baseMPI->mpirank == 0) std::cout << "--- MPI direct sum ---------------" << std::endl;
  for (int irank=0; irank<baseMPI->mpisize; irank++) {
    if (baseMPI->mpirank == 0) std::cout << "Direct loop          : " << irank+1 << "/" << baseMPI->mpisize << std::endl;
    MPI_Shift(x2, Nj3, baseMPI->mpisize, baseMPI->mpirank);
    MPI_Shift(q2, Nj,  baseMPI->mpisize, baseMPI->mpirank);
    for (int i=0; i<Ni; i++) {
      double pp = 0, fx = 0, fy = 0, fz = 0;
      for (int ix=-prange; ix<=prange; ix++) {
        for (int iy=-prange; iy<=prange; iy++) {
          for (int iz=-prange; iz<=prange; iz++) {
            Xperiodic[0] = ix * cycle;
            Xperiodic[1] = iy * cycle;
            Xperiodic[2] = iz * cycle;
            for (int j=0; j<Nj; j++) {
              double dx = x[3*i+0] - x2[3*j+0] - Xperiodic[0];
              double dy = x[3*i+1] - x2[3*j+1] - Xperiodic[1];
              double dz = x[3*i+2] - x2[3*j+2] - Xperiodic[2];
              double R2 = dx * dx + dy * dy + dz * dz;
              double invR = 1 / std::sqrt(R2);
              if (R2 == 0) invR = 0;
              double invR3 = q2[j] * invR * invR * invR;
              pp += q2[j] * invR;
              fx += dx * invR3;
              fy += dy * invR3;
              fz += dz * invR3;
            }
          }
        }
      }
      p[i] += pp;
      f[3*i+0] -= fx;
      f[3*i+1] -= fy;
      f[3*i+2] -= fz;
    }
  }
  double localDipole[3] = {0, 0, 0};
  for (int i=0; i<Ni; i++) {
    for (int d=0; d<3; d++) localDipole[d] += x[3*i+d] * q[i];
  }
  int N;
  MPI_Allreduce(&Ni, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  double globalDipole[3];
  MPI_Allreduce(localDipole, globalDipole, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double norm = 0;
  for (int d=0; d<3; d++) {
    norm += globalDipole[d] * globalDipole[d];
  }
  double coef = 4 * M_PI / (3 * cycle * cycle * cycle);
  for (int i=0; i<Ni; i++) {
    p[i] -= coef * norm / N / q[i];
    f[3*i+0] -= coef * globalDipole[0];
    f[3*i+1] -= coef * globalDipole[1];
    f[3*i+2] -= coef * globalDipole[2];
  }
  delete[] x2;
  delete[] q2;
}


extern "C" void Test_Sum(int Ni, double * p, double * p2,
			 double * q,
			 double * f, double * f2) {
  int mpirank = baseMPI->mpirank;
  int stringLength = 20;
  double potSum = 0, potSum2 = 0, accDif = 0, accNrm = 0;
  for (int i=0; i<Ni; i++) {
    potSum  += p[i]  * q[i];
    potSum2 += p2[i] * q[i];
    accDif  += (f[3*i+0] - f2[3*i+0]) * (f[3*i+0] - f2[3*i+0])
      + (f[3*i+1] - f2[3*i+1]) * (f[3*i+1] - f2[3*i+1])
      + (f[3*i+2] - f2[3*i+2]) * (f[3*i+2] - f2[3*i+2]);
    accNrm  += f2[3*i+0] * f2[3*i+0] + f2[3*i+1] * f2[3*i+1] + f2[3*i+2] * f2[3*i+2];
  }
  double potSumGlob, potSumGlob2, accDifGlob, accNrmGlob;
  MPI_Reduce(&potSum,  &potSumGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&potSum2, &potSumGlob2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&accDif,  &accDifGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&accNrm,  &accNrmGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double potDifGlob = (potSumGlob - potSumGlob2) * (potSumGlob - potSumGlob2);
  double potNrmGlob = potSumGlob * potSumGlob;
  if (mpirank == 0) {
    std::cout << "--- FMM vs. Ewald  ---------------" << std::endl;
    std::cout << std::setw(stringLength) << std::left << std::scientific
  	      << "Rel. L2 Error (pot)" << " : " << std::sqrt(potDifGlob/potNrmGlob) << std::endl;
    std::cout << std::setw(stringLength) << std::left
	      << "Rel. L2 Error (acc)" << " : " << std::sqrt(accDifGlob/accNrmGlob) << std::endl;
  }
}

extern "C" void Test_Direct(int Ni, double * p, double * p2,
			 double * q) {
  int mpirank = get_mpirank();
  int stringLength = 20;
  double potSum = 0, potSum2 = 0, accDif = 0, accNrm = 0;
  for (int i=0; i<Ni; i++) {
    potSum  += p[i]  * q[i];
    potSum2 += p2[i] * q[i];
  }
  double potSumGlob, potSumGlob2;
  MPI_Reduce(&potSum,  &potSumGlob,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&potSum2, &potSumGlob2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double potDif2Glob = (potSumGlob - potSumGlob2) * (potSumGlob - potSumGlob2);
  double potDifGlob = potSumGlob - potSumGlob2;
  if (mpirank == 0) {
    std::cout << "--- FMM vs. Direct  ---------------" << std::endl;
    std::cout << std::setw(stringLength) << std::left << std::scientific
  	      << "Error (potDiff^2)" << " : " << potDif2Glob << std::endl;
    std::cout << std::setw(stringLength) << std::left
	      << "Error (potDiff)" << " : " << potDifGlob << std::endl;
  }
}
