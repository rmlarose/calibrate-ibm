/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for twist-basis parallelization scheme
*/
#ifndef SBD_CHEMISTRY_TPB_MULT_H
#define SBD_CHEMISTRY_TPB_MULT_H

#include <chrono>

namespace sbd {

  // current mult
  template <typename ElemT>
  void mult(const std::vector<ElemT> & hii,
	    const std::vector<std::vector<size_t*>> & ih,
	    const std::vector<std::vector<size_t*>> & jh,
	    const std::vector<std::vector<ElemT*>> & hij,
	    const std::vector<std::vector<size_t>> & len,
	    const std::vector<size_t> & tasktype,
	    const std::vector<size_t> & adetshift,
	    const std::vector<size_t> & bdetshift,
	    const size_t adet_comm_size,
	    const size_t bdet_comm_size,
	    const std::vector<ElemT> & Wk,
	    std::vector<ElemT> & Wb,
	    size_t bit_length,
	    MPI_Comm h_comm,
	    MPI_Comm b_comm,
	    MPI_Comm t_comm) {

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    int mpi_rank_b = 0;
    int mpi_size_b = 1;
    int mpi_rank_t = 0;
    int mpi_size_t = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);
    MPI_Comm_rank(b_comm,&mpi_rank_b);
    MPI_Comm_size(b_comm,&mpi_size_b);
    MPI_Comm_rank(t_comm,&mpi_rank_t);
    MPI_Comm_size(t_comm,&mpi_size_t);

    // distribute vector by t_comm

    auto time_copy_start = std::chrono::high_resolution_clock::now();
    std::vector<ElemT> T(Wk);
    std::vector<ElemT> R(Wk);
    Mpi2dSlide(Wk,T,adet_comm_size,bdet_comm_size,
	       -adetshift[0],-bdetshift[0],b_comm);

    auto time_copy_end = std::chrono::high_resolution_clock::now();

    auto time_mult_start = std::chrono::high_resolution_clock::now();

    size_t num_threads = 1;
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
      if( mpi_rank_t == 0 ) {
#pragma omp for
	for(size_t i=0; i < T.size(); i++) {
	  Wb[i] += hii[i] * T[i];
	}
      }
    }

    for(size_t task=0; task < tasktype.size(); task++) {

#pragma omp parallel
      {
	size_t thread_id = omp_get_thread_num();
	for(size_t k=0; k < len[task][thread_id]; k++) {
	  Wb[ih[task][thread_id][k]] += hij[task][thread_id][k] * T[jh[task][thread_id][k]];
	}
      }


#pragma omp barrier
      if( tasktype[task] == 0 && task != tasktype.size()-1 ) {
	int adetslide = adetshift[task]-adetshift[task+1];
	int bdetslide = bdetshift[task]-bdetshift[task+1];
	R.resize(T.size());
	std::memcpy(R.data(),T.data(),T.size()*sizeof(ElemT));
	Mpi2dSlide(R,T,adet_comm_size,bdet_comm_size,adetslide,bdetslide,b_comm);
      }
    }
    auto time_mult_end = std::chrono::high_resolution_clock::now();

    auto time_comm_start = std::chrono::high_resolution_clock::now();
    MpiAllreduce(Wb,MPI_SUM,t_comm);
    MpiAllreduce(Wb,MPI_SUM,h_comm);
    auto time_comm_end = std::chrono::high_resolution_clock::now();

#ifdef SBD_DEBUG_MULT
    auto time_copy_count = std::chrono::duration_cast<std::chrono::microseconds>(time_copy_end-time_copy_start).count();
    auto time_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_mult_end-time_mult_start).count();
    auto time_comm_count = std::chrono::duration_cast<std::chrono::microseconds>(time_comm_end-time_comm_start).count();
    double time_copy = 1.0e-6 * time_copy_count;
    double time_mult = 1.0e-6 * time_mult_count;
    double time_comm = 1.0e-6 * time_comm_count;
    std::cout << " mult: time for first copy     = " << time_copy << std::endl;
    std::cout << " mult: time for multiplication = " << time_mult << std::endl;
    std::cout << " mult: time for allreduce comm = " << time_comm << std::endl;
#endif
  }


#ifdef SBD_TRADMODE
  template <typename ElemT>
  void mult(const std::vector<ElemT> & hii,
	    const std::vector<ElemT> & Wk,
	    std::vector<ElemT> & Wb,
	    const std::vector<std::vector<size_t>> & adets,
	    const std::vector<std::vector<size_t>> & bdets,
	    const size_t bit_length,
	    const size_t norbs,
	    const size_t adet_comm_size,
	    const size_t bdet_comm_size,
	    const std::vector<TaskHelpers> & helper,
	    const ElemT & I0,
	    const oneInt<ElemT> & I1,
	    const twoInt<ElemT> & I2,
	    MPI_Comm h_comm,
	    MPI_Comm b_comm,
	    MPI_Comm t_comm) {

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);

    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    size_t braAlphaSize = 0;
    size_t braBetaSize  = 0;
    if( helper.size() != 0 ) {
      braAlphaSize = helper[0].braAlphaEnd-helper[0].braAlphaStart;
      braBetaSize  = helper[0].braBetaEnd-helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = adets.size();
    size_t bdet_min = 0;
    size_t bdet_max = bdets.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    auto time_copy_start = std::chrono::high_resolution_clock::now();
    std::vector<ElemT> T;
    std::vector<ElemT> R;
    T.reserve(max_det_size);
    R.reserve(max_det_size);
    if( helper.size() != 0 ) {
      Mpi2dSlide(Wk,T,adet_comm_size,bdet_comm_size,
		 -helper[0].adetShift,-helper[0].bdetShift,b_comm);
    }
    auto time_copy_end = std::chrono::high_resolution_clock::now();

    auto time_mult_start = std::chrono::high_resolution_clock::now();

    if( mpi_rank_t == 0 ) {
#pragma omp parallel for
       for(size_t i=0; i < T.size(); i++) {
          Wb[i] += hii[i] * T[i];
       }
    }

#ifdef SBD_DEBUG_MULT
       std::cout << " End multiplication of diagonal term at mpi process (h,b,t) = ("
	      << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << ")" << std::endl;
#endif

    double time_slid = 0.0;
    for(size_t task=0; task < helper.size(); task++) {

#ifdef SBD_DEBUG_MULT
      std::cout << " Start multiplication for task " << task << " at (h,b,t) = ("
		<< mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << "): task type = "
		<< helper[task].taskType << ", bra-adet range = ["
		<< helper[task].braAlphaStart << "," << helper[task].braAlphaEnd << "), bra-bdet range = ["
		<< helper[task].braBetaStart << "," << helper[task].braBetaEnd << "), ket-adet range = ["
		<< helper[task].ketAlphaStart << "," << helper[task].ketAlphaEnd << "), ket-bdet range = ["
		<< helper[task].ketBetaStart << "," << helper[task].ketBetaEnd << "), ket wf =";
      for(size_t i=0; i < std::min(static_cast<size_t>(4),T.size()); i++) {
	std::cout << " " << T[i];
      }
      std::cout << std::endl;
#endif
      size_t ketAlphaSize = helper[task].ketAlphaEnd-helper[task].ketAlphaStart;
      size_t ketBetaSize  = helper[task].ketBetaEnd-helper[task].ketBetaStart;
#pragma omp parallel
      {
	size_t ia_start = helper[task].braAlphaStart;
	size_t ia_end   = helper[task].braAlphaEnd;

	auto DetI = DetFromAlphaBeta(adets[0],bdets[0],bit_length,norbs);
	std::vector<int> c(2,0);
	std::vector<int> d(2,0);

	if( helper[task].taskType == 2 ) { // beta range are same
#pragma omp for	schedule(dynamic)
	  for(size_t ia = ia_start; ia < ia_end; ia++) {
	    for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {

	      size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		+ib-helper[task].braBetaStart;
	      if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;

	      DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,DetI);

	      // single alpha excitation
	      for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                +ib-helper[task].ketBetaStart;
		ElemT eij = OneExcite(DetI,bit_length,
				      helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
				      helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }
	      // double alpha excitation
	      for(size_t j=0; j < helper[task].DoublesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].DoublesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		               + ib-helper[task].ketBetaStart;
		ElemT eij = TwoExcite(DetI,bit_length,
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+0],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+1],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+2],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+3],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }

	    } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	  } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)

	} else if ( helper[task].taskType == 1 ) { // alpha range are same
#pragma omp for schedule(dynamic)
	  for(size_t ia = ia_start; ia < ia_end; ia++) {
	    for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {

	      size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		              +ib-helper[task].braBetaStart;
	      if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
	      DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,DetI);

	      // single beta excitation
	      for(size_t j=0; j < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][j];
		size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		               + jb-helper[task].ketBetaStart;
		ElemT eij = OneExcite(DetI,bit_length,
				      helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+0],
				      helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+1],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }
	      // double beta excitation
	      for(size_t j=0; j < helper[task].DoublesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		size_t jb = helper[task].DoublesFromBetaSM[ib-helper[task].braBetaStart][j];
		size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		               + jb-helper[task].ketBetaStart;
		ElemT eij = TwoExcite(DetI,bit_length,
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+0],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+1],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+2],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+3],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }
	    } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	  } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)


	} else {
#pragma omp for schedule(dynamic)
	  for(size_t ia = ia_start; ia < ia_end; ia++) {
	    for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {

	      size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		              +ib-helper[task].braBetaStart;
	      if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;

	      DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,DetI);

	      // two-particle excitation composed of single alpha and single beta
	      for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		for(size_t k=0; k < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; k++) {
		  size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][k];
		  size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                  +jb-helper[task].ketBetaStart;
		  ElemT eij = TwoExcite(DetI,bit_length,
					helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
					helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+0],
					helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
					helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+1],
					I1,I2);
		  Wb[braIdx] += eij * T[ketIdx];
		}
	      }

	    } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	  } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	} // if ( helper[task].taskType == ? )
      } // end pragma paralell

      if( helper[task].taskType == 0 && task != helper.size()-1 ) {
#ifdef SBD_DEBUG_MULT
	size_t adet_rank = mpi_rank_b / bdet_comm_size;
	size_t bdet_rank = mpi_rank_b % bdet_comm_size;
	size_t adet_rank_task = (adet_rank+helper[task].adetShift) % adet_comm_size;
	size_t bdet_rank_task = (bdet_rank+helper[task].bdetShift) % bdet_comm_size;
	size_t adet_rank_next = (adet_rank+helper[task+1].adetShift) % adet_comm_size;
	size_t bdet_rank_next = (bdet_rank+helper[task+1].bdetShift) % bdet_comm_size;
	std::cout << " mult: task " << task << " at mpi process (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t
		  << "): two-dimensional slide communication from ("
		  << adet_rank_task << "," << bdet_rank_task << ") to ("
		  << adet_rank_next << "," << bdet_rank_next << ")"
		  << std::endl;

#endif
	int adetslide = helper[task].adetShift-helper[task+1].adetShift;
	int bdetslide = helper[task].bdetShift-helper[task+1].bdetShift;
	R.resize(T.size());
	std::memcpy(R.data(),T.data(),T.size()*sizeof(ElemT));
	auto time_slid_start = std::chrono::high_resolution_clock::now();
	Mpi2dSlide(R,T,adet_comm_size,bdet_comm_size,adetslide,bdetslide,b_comm);
	auto time_slid_end = std::chrono::high_resolution_clock::now();
	auto time_slid_count = std::chrono::duration_cast<std::chrono::microseconds>(time_slid_end-time_slid_start).count();
	time_slid += 1.0e-6 * time_slid_count;
      }

    } // end for(size_t task=0; task < helper.size(); task++)
    auto time_mult_end = std::chrono::high_resolution_clock::now();

    auto time_comm_start = std::chrono::high_resolution_clock::now();
    MpiAllreduce(Wb,MPI_SUM,t_comm);
    MpiAllreduce(Wb,MPI_SUM,h_comm);
    auto time_comm_end = std::chrono::high_resolution_clock::now();

#ifdef SBD_DEBUG_MULT
    auto time_copy_count = std::chrono::duration_cast<std::chrono::microseconds>(time_copy_end-time_copy_start).count();
    auto time_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_mult_end-time_mult_start).count();
    auto time_comm_count = std::chrono::duration_cast<std::chrono::microseconds>(time_comm_end-time_comm_start).count();

    double time_copy = 1.0e-6 * time_copy_count;
    double time_mult = 1.0e-6 * time_mult_count;
    double time_comm = 1.0e-6 * time_comm_count;
    std::cout << " mult: time for first copy     = " << time_copy << std::endl;
    std::cout << " mult: time for multiplication = " << time_mult << std::endl;
    std::cout << " mult: time for 2d slide comm  = " << time_slid << std::endl;
    std::cout << " mult: time for allreduce comm = " << time_comm << std::endl;
#endif

  } // end function

#else

  template <typename ElemT>
  void mult(const std::vector<ElemT> & hii,
	    const std::vector<ElemT> & Wk,
	    std::vector<ElemT> & Wb,
	    const std::vector<std::vector<size_t>> & adets,
	    const std::vector<std::vector<size_t>> & bdets,
	    const size_t bit_length,
	    const size_t norbs,
	    const size_t adet_comm_size,
	    const size_t bdet_comm_size,
	    const std::vector<TaskHelpers> & helper,
	    const ElemT & I0,
	    const oneInt<ElemT> & I1,
	    const twoInt<ElemT> & I2,
	    MPI_Comm h_comm,
	    MPI_Comm b_comm,
	    MPI_Comm t_comm) {

#ifdef SBD_DEBUG_TUNING
    std::cout << " multiplication with round-robin assignment of work to OpenMP threads " << std::endl;
#endif

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);

    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    size_t braAlphaSize = 0;
    size_t braBetaSize  = 0;
    if( helper.size() != 0 ) {
      braAlphaSize = helper[0].braAlphaEnd-helper[0].braAlphaStart;
      braBetaSize  = helper[0].braBetaEnd-helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = adets.size();
    size_t bdet_min = 0;
    size_t bdet_max = bdets.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    int num_threads = 1;

    auto time_copy_start = std::chrono::high_resolution_clock::now();
    std::vector<ElemT> T;
    std::vector<ElemT> R;
    T.reserve(max_det_size);
    R.reserve(max_det_size);
    if( helper.size() != 0 ) {
      Mpi2dSlide(Wk,T,adet_comm_size,bdet_comm_size,
		 -helper[0].adetShift,-helper[0].bdetShift,b_comm);
    }
    auto time_copy_end = std::chrono::high_resolution_clock::now();

    auto time_mult_start = std::chrono::high_resolution_clock::now();

    num_threads = omp_get_max_threads();

    if( mpi_rank_t == 0 ) {
#pragma omp parallel for
       for(size_t i=0; i < T.size(); i++) {
          Wb[i] += hii[i] * T[i];
       }
    }

#ifdef SBD_DEBUG_MULT
    std::cout << " End multiplication of diagonal term at mpi process (h,b,t) = ("
	      << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << ")" << std::endl;
#endif

    double time_slid = 0.0;
    for(size_t task=0; task < helper.size(); task++) {

#ifdef SBD_DEBUG_MULT
      std::cout << " Start multiplication for task " << task << " at (h,b,t) = ("
		<< mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << "): task type = "
		<< helper[task].taskType << ", bra-adet range = ["
		<< helper[task].braAlphaStart << "," << helper[task].braAlphaEnd << "), bra-bdet range = ["
		<< helper[task].braBetaStart << "," << helper[task].braBetaEnd << "), ket-adet range = ["
		<< helper[task].ketAlphaStart << "," << helper[task].ketAlphaEnd << "), ket-bdet range = ["
		<< helper[task].ketBetaStart << "," << helper[task].ketBetaEnd << "), ket wf =";
      for(size_t i=0; i < std::min(static_cast<size_t>(4),T.size()); i++) {
	std::cout << " " << T[i];
      }
      std::cout << std::endl;
#endif
      size_t ketAlphaSize = helper[task].ketAlphaEnd-helper[task].ketAlphaStart;
      size_t ketBetaSize  = helper[task].ketBetaEnd-helper[task].ketBetaStart;
#pragma omp parallel
      {
        // round-robin assignment of work to threads
	size_t thread_id = omp_get_thread_num();
	size_t ia_start = thread_id + helper[task].braAlphaStart;
	size_t ia_end   = helper[task].braAlphaEnd;

	auto DetI = DetFromAlphaBeta(adets[0],bdets[0],bit_length,norbs);
	std::vector<int> c(2,0);
	std::vector<int> d(2,0);
	
	if( helper[task].taskType == 2 ) { // beta range are same
	  for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	    for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
	      
	      size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		+ib-helper[task].braBetaStart;
	      if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
	      
	      DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,DetI);
	      
	      // single alpha excitation
	      for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		  +ib-helper[task].ketBetaStart;
		ElemT eij = OneExcite(DetI,bit_length,
				      helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
				      helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }
	      // double alpha excitation
	      for(size_t j=0; j < helper[task].DoublesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].DoublesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		  + ib-helper[task].ketBetaStart;
		ElemT eij = TwoExcite(DetI,bit_length,
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+0],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+1],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+2],
				      helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+3],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }
	    } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	  } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	} else if ( helper[task].taskType == 1 ) { // alpha range are same
	  
	  for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	    for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
	      size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		+ib-helper[task].braBetaStart;
	      if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
	      
	      DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,DetI);
	      
	      // single beta excitation
	      for(size_t j=0; j < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][j];
		size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		  + jb-helper[task].ketBetaStart;
		ElemT eij = OneExcite(DetI,bit_length,
				      helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+0],
				      helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+1],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }
	      // double beta excitation
	      for(size_t j=0; j < helper[task].DoublesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		size_t jb = helper[task].DoublesFromBetaSM[ib-helper[task].braBetaStart][j];
		size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		  + jb-helper[task].ketBetaStart;
		ElemT eij = TwoExcite(DetI,bit_length,
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+0],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+1],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+2],
				      helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+3],
				      I1,I2);
		Wb[braIdx] += eij * T[ketIdx];
	      }
	    } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	  } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	} else {
	  for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	    for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
	      size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		+ib-helper[task].braBetaStart;
	      if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
	      
	      DetFromAlphaBeta(adets[ia],bdets[ib],bit_length,norbs,DetI);
	      
	      // two-particle excitation composed of single alpha and single beta
	      for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		for(size_t k=0; k < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; k++) {
		  size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][k];
		  size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		    +jb-helper[task].ketBetaStart;
		  ElemT eij = TwoExcite(DetI,bit_length,
					helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
					helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+0],
					helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
					helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+1],
					I1,I2);
		  Wb[braIdx] += eij * T[ketIdx];
		}
	      }
	    } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	  } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	} // if ( helper[task].taskType == ? )
	
      } // end pragma paralell
      
      if( helper[task].taskType == 0 && task != helper.size()-1 ) {
#ifdef SBD_DEBUG_MULT
	size_t adet_rank = mpi_rank_b / bdet_comm_size;
	size_t bdet_rank = mpi_rank_b % bdet_comm_size;
	size_t adet_rank_task = (adet_rank+helper[task].adetShift) % adet_comm_size;
	size_t bdet_rank_task = (bdet_rank+helper[task].bdetShift) % bdet_comm_size;
	size_t adet_rank_next = (adet_rank+helper[task+1].adetShift) % adet_comm_size;
	size_t bdet_rank_next = (bdet_rank+helper[task+1].bdetShift) % bdet_comm_size;
	std::cout << " mult: task " << task << " at mpi process (h,b,t) = ("
		  << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t
		  << "): two-dimensional slide communication from ("
		  << adet_rank_task << "," << bdet_rank_task << ") to ("
		  << adet_rank_next << "," << bdet_rank_next << ")"
		  << std::endl;

#endif
	int adetslide = helper[task].adetShift-helper[task+1].adetShift;
	int bdetslide = helper[task].bdetShift-helper[task+1].bdetShift;
	R.resize(T.size());
	std::memcpy(R.data(),T.data(),T.size()*sizeof(ElemT));
	auto time_slid_start = std::chrono::high_resolution_clock::now();
	Mpi2dSlide(R,T,adet_comm_size,bdet_comm_size,adetslide,bdetslide,b_comm);
	auto time_slid_end = std::chrono::high_resolution_clock::now();
	auto time_slid_count = std::chrono::duration_cast<std::chrono::microseconds>(time_slid_end-time_slid_start).count();
	time_slid += 1.0e-6 * time_slid_count;
      }

    } // end for(size_t task=0; task < helper.size(); task++)
    auto time_mult_end = std::chrono::high_resolution_clock::now();

    auto time_comm_start = std::chrono::high_resolution_clock::now();
    MpiAllreduce(Wb,MPI_SUM,t_comm);
    MpiAllreduce(Wb,MPI_SUM,h_comm);
    auto time_comm_end = std::chrono::high_resolution_clock::now();

#ifdef SBD_DEBUG_MULT
    auto time_copy_count = std::chrono::duration_cast<std::chrono::microseconds>(time_copy_end-time_copy_start).count();
    auto time_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_mult_end-time_mult_start).count();
    auto time_comm_count = std::chrono::duration_cast<std::chrono::microseconds>(time_comm_end-time_comm_start).count();

    double time_copy = 1.0e-6 * time_copy_count;
    double time_mult = 1.0e-6 * time_mult_count;
    double time_comm = 1.0e-6 * time_comm_count;
    std::cout << " mult: time for first copy     = " << time_copy << std::endl;
    std::cout << " mult: time for multiplication = " << time_mult << std::endl;
    std::cout << " mult: time for 2d slide comm  = " << time_slid << std::endl;
    std::cout << " mult: time for allreduce comm = " << time_comm << std::endl;
#endif

  } // end function

#endif // SBD_TRADMODE

}

#endif
