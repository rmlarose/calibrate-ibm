/**
@file sbd/chemistry/gdb/carryover.h
@brief functions for carryover bitstrings for gdb
*/
#ifndef SBD_CHEMISTRY_GDB_CARRYOVER_H
#define SBD_CHEMISTRY_GDB_CARRYOVER_H

namespace sbd {
  namespace gdb {

    template <typename ElemT, typename RealT>
    void CarryOverDet(const std::vector<ElemT> & w,
		      const std::vector<std::vector<size_t>> & det,
		      MPI_Comm b_comm,
		      size_t kept,
		      std::vector<std::vector<size_t>> & rdet,
		      RealT & discarted_weight) {
      // using RealT = typename GetRealType<ElemT>::RealT;
      std::vector<RealT> r(w.size());
#pragma omp parallel for
      for(size_t i=0; i < w.size(); i++) {
	r[i] = Conjugate(w[i])*w[i];
      }
      std::vector<size_t> ranking(r.size());
      mpi_find_ranking(r,ranking,b_comm);
      size_t rdet_size = 0;
      size_t num_threads = omp_get_max_threads();
      std::vector<size_t> local_size(num_threads,0);
#pragma omp parallel
      {
	size_t thread_id = omp_get_thread_num();
	for(size_t k=thread_id; k < det.size(); k+=num_threads) {
	  if( ranking[k] < kept ) {
	    local_size[thread_id]++;
	  }
	}
      }
      std::vector<size_t> offset(num_threads,0);
      for(size_t tid=0; tid < num_threads; tid++) {
	rdet_size += local_size[tid];
	if( tid < num_threads-1 ) {
	  offset[tid+1] = rdet_size;
	}
      }
      rdet.resize(rdet_size);
      std::vector<RealT> keep_weight_local(num_threads,0.0);
#pragma omp parallel
      {
	size_t thread_id = omp_get_thread_num();
	size_t local_addr = offset[thread_id];
	for(size_t k=thread_id; k < det.size(); k+=num_threads) {
	  if( ranking[k] < kept ) {
	    rdet[local_addr++] = det[k];
	    keep_weight_local[thread_id] += r[k];
	  }
	}
      }
      RealT keep_weight = 0.0;
      for(size_t tid=0; tid < num_threads; tid++) {
	keep_weight += keep_weight_local[tid];
      }
      RealT keep_weight_global = 0.0;
      MPI_Datatype DataT = GetMpiType<RealT>::MpiT;
      MPI_Allreduce(&keep_weight,&keep_weight_global,
		    1,DataT,MPI_SUM,b_comm);
      discarted_weight = 1.0-keep_weight_global;
      sort_bitarray(rdet);
    }
		      
    
  }
}

#endif
