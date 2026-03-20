// This is a part of qsbd
/**
@file /sbd/framework/dm_vector.h
@brief function for vector on distributed-memory
*/

#ifndef SBD_FRAMEWORK_DM_VECTOR_H
#define SBD_FRAMEWORK_DM_VECTOR_H

#define SBD_MAX_THREADS 192

#include <random>

namespace sbd {

  template <typename ElemT>
  void Zero(std::vector<ElemT> & W) {
    size_t w_size = W.size();
    W = std::vector<ElemT>(w_size,ElemT(0.0));
  }

  template <typename ElemT>
  void InnerProduct(const std::vector<ElemT> & X,
		    const std::vector<ElemT> & Y,
		    ElemT & res,
		    MPI_Comm comm) {
    int nth = omp_get_max_threads();
    ElemT array[SBD_MAX_THREADS];
    ElemT sum = ElemT(0.0);
// use Kahan summation for each thread and add the private sums in deterministic order
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      ElemT mysum = ElemT(0.0);
      ElemT eps   = ElemT(0.0);
      ElemT val, tmp;
      #pragma omp for schedule(static)
      for(size_t is=0; is < X.size(); is++) {
        val = Conjugate(X[is]) * Y[is] - eps;
        tmp = mysum + val;
        eps = (tmp - mysum) - val;
        mysum = tmp;
      }
      array[tid] = mysum;
    }
    for (int i = 0; i < nth; i++) sum += array[i];

    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
    MPI_Allreduce(&sum,&res,1,DataT,MPI_SUM,comm);
  }

  template <typename ElemT, typename RealT>
  void Normalize(std::vector<ElemT> & X,
		 RealT & res,
		 MPI_Comm comm) {
    res = 0.0;
    RealT sum = 0.0;
    RealT array[SBD_MAX_THREADS];
    int nth = omp_get_max_threads();
// use Kahan summation for each thread and add the private sums in deterministic order
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      RealT mysum = 0.0;
      RealT eps   = 0.0;
      RealT val, tmp;
      #pragma omp for schedule(static)
      for(size_t is=0; is < X.size(); is++) {
        val = GetReal( Conjugate(X[is]) * X[is] ) - eps;
        tmp = mysum + val;
        eps = (tmp - mysum) - val;
        mysum = tmp;
      }
      array[tid] = mysum;
    }
    for (int i = 0; i < nth; i++) sum += array[i];
    
    MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
    MPI_Allreduce(&sum,&res,1,DataT,MPI_SUM,comm);
    res = std::sqrt(res);
    ElemT factor = ElemT(1.0/res);
#pragma omp parallel for schedule(static)
    for(size_t is=0; is < X.size(); is++) {
      X[is] *= factor;
    }
  }

  template <typename ElemT>
  void apply_dist(ElemT & x, std::mt19937 & gen) {
    std::uniform_real_distribution<double> dist(-1,1);
    x = ElemT(dist(gen));
  }

  template<>
  void apply_dist(std::complex<double> & x, std::mt19937 & gen) {
    std::uniform_real_distribution<double> dist(-1,1);
    x = std::complex<double>(dist(gen),dist(gen));
  }

  template <typename ElemT>
  void Randomize(std::vector<ElemT> & X,
		 MPI_Comm b_comm,
		 MPI_Comm h_comm) {
    using RealT = typename GetRealType<ElemT>::RealT;
    int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int nth = omp_get_max_threads();
    RealT array[SBD_MAX_THREADS];
    RealT sum=0.0;
// use Kahan summation for each thread and add the private sums in deterministic order
    #pragma omp parallel
    {
      unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
      std::mt19937 gen(seed);
      std::uniform_real_distribution<RealT> dist(-1,1);
      int tid = omp_get_thread_num();
      RealT mysum = 0.0;
      RealT eps   = 0.0;
      RealT val, tmp;
      #pragma omp for schedule(static)
      for(size_t is=0; is < X.size(); is++) {
	apply_dist(X[is],gen);
        val = GetReal( Conjugate(X[is]) * X[is] ) - eps;
        tmp = mysum + val;
        eps = (tmp - mysum) - val;
        mysum = tmp;
      }
      array[tid] = mysum;
    }
    for (int i = 0; i < nth; i++) sum += array[i];

    RealT res;
    MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
    if( mpi_size_b != 1 ) {
      MPI_Allreduce(&sum,&res,1,DataT,MPI_SUM,b_comm);
    } else {
      res = sum;
    }
    res = std::sqrt(res);
    ElemT factor = ElemT(1.0/res);
#pragma omp parallel for
    for(size_t is=0; is < X.size(); is++) {
      X[is] *= factor;
    }
    MPI_Datatype MpiDataT = GetMpiType<ElemT>::MpiT;
    if( mpi_size_h != 1 ) {
      MPI_Bcast(X.data(),X.size(),MpiDataT,0,h_comm);
    }
  }

  template <typename ElemT>
  void Randomize(std::vector<ElemT> & X,
		 MPI_Comm b_comm) {
    using RealT = typename GetRealType<ElemT>::RealT;
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int nth = omp_get_max_threads();
    RealT array[SBD_MAX_THREADS];
    RealT sum=0.0;
// use Kahan summation for each thread and add the private sums in deterministic order
    #pragma omp parallel
    {
      unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
      std::mt19937 gen(seed);
      std::uniform_real_distribution<RealT> dist(-1,1);
      int tid = omp_get_thread_num();
      RealT mysum = 0.0;
      RealT eps   = 0.0;
      RealT val, tmp;
      #pragma omp for schedule(static)
      for(size_t is=0; is < X.size(); is++) {
	apply_dist(X[is],gen);
        val = GetReal( Conjugate(X[is]) * X[is] ) - eps;
        tmp = mysum + val;
        eps = (tmp - mysum) - val;
        mysum = tmp;
      }
      array[tid] = mysum;
    }
    for (int i = 0; i < nth; i++) sum += array[i];

    RealT res;
    MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
    if( mpi_size_b != 1 ) {
      MPI_Allreduce(&sum,&res,1,DataT,MPI_SUM,b_comm);
    } else {
      res = sum;
    }
    res = std::sqrt(res);
    ElemT factor = ElemT(1.0/res);
#pragma omp parallel for
    for(size_t is=0; is < X.size(); is++) {
      X[is] *= factor;
    }
  }
  
  template <typename ElemT>
  void Swap(ElemT a, std::vector<ElemT> & X,
	    ElemT b, std::vector<ElemT> & Y) {
#pragma omp parallel
    {
      ElemT c;
#pragma omp for
      for(size_t is=0; is < X.size(); is++) {
	c = a * X[is];
	X[is] = b * Y[is];
	Y[is] = c;
      }
    }
  }

  template <typename ElemT, typename IntT>
  void MGS(const std::vector<std::vector<ElemT>> & V,
	   IntT k,
	   std::vector<ElemT> & W,
	   std::vector<ElemT> & h,
	   MPI_Comm comm) {
    h.resize(k);
    for(IntT i=0; i < k; i++) {
      InnerProduct(V[i],W,h[i],comm);
      for(size_t s=0; s < W.size(); s++) {
	W[s] -= h[i] * V[i][s];
      }
    }
  }
	   
  
}

#endif
