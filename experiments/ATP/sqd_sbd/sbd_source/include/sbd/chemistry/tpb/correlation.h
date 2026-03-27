/**
@file sbd/chemistry/tpb/correlation.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/
#ifndef SBD_CHEMISTRY_TPB_CORRELATION_H
#define SBD_CHEMISTRY_TPB_CORRELATION_H

namespace sbd {


  /*
  template <typename ElemT>
  void ZeroDiffCorrelation(const std::vector<size_t> & DetI,
			   ElemT WeightI,
			   size_t bit_length,
			   size_t norb,
			   std::vector<std::vector<ElemT>> & onebody,
			   std::vector<std::vector<ElemT>> & twobody) {
    std::vector<int> closed;
    int num_closed = getClosed(DetI,bit_length,2*norb,closed);

    for(int i=0; i < num_closed; i++) {
      int oi = closed.at(i)/2;
      int si = closed.at(i)%2;
      onebody[si][oi+norb*oi] += Conjugate(WeightI)*WeightI;
      for(int j=i+1; j < num_closed; j++) {
	int oj = closed.at(j)/2;
	int sj = closed.at(j)%2;
	twobody[si+2*sj][oi+norb*oj+norb*norb*oi+norb*norb*norb*oj]
	  += Conjugate(WeightI) * WeightI;
	twobody[sj+2*si][oj+norb*oi+norb*norb*oj+norb*norb*norb*oi]
	  += Conjugate(WeightI) * WeightI;
	if( si == sj ) {
	  twobody[si+2*sj][oi+norb*oj+norb*norb*oj+norb*norb*norb*oi]
	    += -Conjugate(WeightI) * WeightI;
	  twobody[sj+2*si][oj+norb*oi+norb*norb*oi+norb*norb*norb*oj]
	    += -Conjugate(WeightI) * WeightI;
	}
      }
    }
  }

  template <typename ElemT>
  void OneDiffCorrelation(const std::vector<size_t> & DetI,
			  const ElemT WeightI,
			  const ElemT WeightJ,
			  const size_t bit_length,
			  const size_t norb,
			  int i,
			  int a,
			  std::vector<std::vector<ElemT>> & onebody,
			  std::vector<std::vector<ElemT>> & twobody) {
    double sgn = 1.0;
    parity(DetI,bit_length,std::min(i,a),std::max(i,a),sgn);
    int oi = i / 2;
    int si = i % 2;
    int oa = a / 2;
    int sa = a % 2;
    onebody[si][oi+norb*oa] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
    size_t one = 1;
    for(int x=0; x < DetI.size(); x++) {
      size_t bits = DetI[x];
      while(bits != 0) {
	int pos = __builtin_ffsl(bits);
	int soj = x * bit_length + pos - 1;
	int oj = soj / 2;
	int sj = soj % 2;

	twobody[si+2*sj][oa+oj*norb+oi*norb*norb+oj*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
	twobody[sj+2*si][oj+oa*norb+oj*norb*norb+oi*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(sgn);
	
	if( si == sj ) {
	  twobody[si+2*sj][oa+oj*norb+oj*norb*norb+oi*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(-sgn);
	  twobody[sj+2*si][oj+oa*norb+oi*norb*norb+oj*norb*norb*norb] += Conjugate(WeightI) * WeightJ * ElemT(-sgn);
	}
	
	bits &= ~(one << (pos-1));
      }
    }
  }

  template <typename ElemT>
  void TwoDiffCorrelation(const std::vector<size_t> & DetI,
			  const ElemT WeightI,
			  const ElemT WeightJ,
			  const size_t bit_length,
			  const size_t norb,
			  int i,
			  int j,
			  int a,
			  int b,
			  std::vector<std::vector<ElemT>> & onebody,
			  std::vector<std::vector<ElemT>> & twobody) {
    double sgn = 1.0;
    int I = std::min(i,j);
    int J = std::max(i,j);
    int A = std::min(a,b);
    int B = std::max(a,b);
    parity(DetI,bit_length,std::min(I,A),std::max(I,A),sgn);
    parity(DetI,bit_length,std::min(J,B),std::max(J,B),sgn);
    if( A > J || B < I ) sgn *= -1.0;
    int oi = I / 2;
    int si = I % 2;
    int oa = A / 2;
    int sa = A % 2;
    int oj = J / 2;
    int sj = J % 2;
    int ob = B / 2;
    int sb = B % 2;

    if( si == sa ) {
      twobody[si+2*sj][oa+norb*ob+norb*norb*(oi+norb*oj)] += ElemT(sgn) * Conjugate(WeightI) * WeightJ;
      twobody[sj+2*si][ob+norb*oa+norb*norb*(oj+norb*oi)] += ElemT(sgn) * Conjugate(WeightI) * WeightJ;
    }

    if( si == sb ) {
      twobody[si+2*sj][oa+norb*ob+norb*norb*(oj+norb*oi)] += ElemT(-sgn) * Conjugate(WeightI) * WeightJ;
      twobody[sj+2*si][ob+norb*oa+norb*norb*(oi+norb*oj)] += ElemT(-sgn) * Conjugate(WeightI) * WeightJ;
    }

  }

  template <typename ElemT>
  void CorrelationTermAddition(const std::vector<size_t> & DetI,
			       const std::vector<size_t> & DetJ,
			       const ElemT WeightI,
			       const ElemT WeightJ,
			       const size_t bit_length,
			       const size_t norb,
			       std::vector<int> & c,
			       std::vector<int> & d,
			       std::vector<std::vector<ElemT>> & onebody,
			       std::vector<std::vector<ElemT>> & twobody) {
    size_t nc = 0;
    size_t nd = 0;

    size_t full_words = (2*norb) / bit_length;
    size_t remaining_bits = (2*norb) % bit_length;

    for(size_t i=0; i < full_words; ++i) {
      size_t diff_c = DetI[i] & ~DetJ[i];
      size_t diff_d = DetJ[i] & ~DetI[i];
      for(size_t bit_pos=0; bit_pos < bit_length; ++bit_pos) {
	if( diff_c & (static_cast<size_t>(1) << bit_pos)) {
	  c[nc] = i*bit_length+bit_pos;
	  nc++;
	}
	if( diff_d & (static_cast<size_t>(1) << bit_pos)) {
	  d[nd] = i*bit_length+bit_pos;
	  nd++;
	}
      }
    }
    if ( remaining_bits > 0 ) {
      size_t mask = (static_cast<size_t>(1) << remaining_bits) -1;
      size_t diff_c = (DetI[full_words] & ~DetJ[full_words]) & mask;
      size_t diff_d = (DetJ[full_words] & ~DetI[full_words]) & mask;
      for(size_t bit_pos = 0; bit_pos < remaining_bits; ++bit_pos) {
	if( diff_c & (static_cast<size_t>(1) << bit_pos) ) {
	  c[nc] = bit_length*full_words+bit_pos;
	  nc++;
	}
	if( diff_d & (static_cast<size_t>(1) << bit_pos) ) {
	  d[nd] = bit_length*full_words+bit_pos;
	  nd++;
	}
      }
    }

    if( nc == 0 ) {
      ZeroDiffCorrelation(DetI,WeightI,bit_length,norb,onebody,twobody);
    } else if ( nc == 1 ) {
      OneDiffCorrelation(DetI,WeightI,WeightJ,bit_length,norb,c[0],d[0],onebody,twobody);
    } else if ( nc == 2 ) {
      TwoDiffCorrelation(DetI,WeightI,WeightJ,bit_length,norb,c[0],c[1],d[0],d[1],onebody,twobody);
    }
  }
  */

  /**
     Function to evaluate the two-particle correlation functions
   */
  template <typename ElemT>
  void Correlation(const std::vector<ElemT> & W,
		   const std::vector<std::vector<size_t>> & adet,
		   const std::vector<std::vector<size_t>> & bdet,
		   const size_t bit_length,
		   const size_t norb,
		   const size_t adet_comm_size,
		   const size_t bdet_comm_size,
		   const std::vector<TaskHelpers> & helper,
		   MPI_Comm h_comm,
		   MPI_Comm b_comm,
		   MPI_Comm t_comm,
		   std::vector<std::vector<ElemT>> & onebody,
		   std::vector<std::vector<ElemT>> & twobody) {

    onebody.resize(2);
    twobody.resize(4);
    onebody[0].resize(norb*norb,ElemT(0.0));
    onebody[1].resize(norb*norb,ElemT(0.0));
    twobody[0].resize(norb*norb*norb*norb,ElemT(0.0));
    twobody[1].resize(norb*norb*norb*norb,ElemT(0.0));
    twobody[2].resize(norb*norb*norb*norb,ElemT(0.0));
    twobody[3].resize(norb*norb*norb*norb,ElemT(0.0));

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
      braAlphaSize = helper[0].braAlphaEnd - helper[0].braAlphaStart;
      braBetaSize  = helper[0].braBetaEnd  - helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = adet.size();
    size_t bdet_min = 0;
    size_t bdet_max = bdet.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    std::vector<ElemT> T;
    std::vector<ElemT> R;
    T.reserve(max_det_size);
    R.reserve(max_det_size);
    if( helper.size() != 0 ) {
      Mpi2dSlide(W,T,adet_comm_size,bdet_comm_size,
		 -helper[0].adetShift,-helper[0].bdetShift,b_comm);
    }

    size_t array_size = (2*norb + bit_length - 1 ) / bit_length;

    size_t num_threads = 1;
    num_threads = omp_get_max_threads();
    
    std::vector<std::vector<std::vector<ElemT>>> onebody_t(num_threads,onebody);
    std::vector<std::vector<std::vector<ElemT>>> twobody_t(num_threads,twobody);

    
    if( mpi_rank_t == 0 ) {
#pragma omp parallel
      {
        // round-robin assignment of work to threads
        size_t thread_id = omp_get_thread_num();
        size_t ia_start = thread_id + helper[0].braAlphaStart;
        size_t ia_end   = helper[0].braAlphaEnd;
        std::vector<size_t> DetT(array_size);
        for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
          for(size_t ib = helper[0].braBetaStart; ib < helper[0].braBetaEnd; ib++) {
            size_t i = (ia - helper[0].braAlphaStart) * braBetaSize
                     +  ib - helper[0].braBetaStart;
            if( ( i % mpi_size_h ) == mpi_rank_h ) {
              DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetT);
              ZeroDiffCorrelation(DetT,W[i],bit_length,norb,onebody_t[thread_id],twobody_t[thread_id]);
            }
          }
        }
      }
    }

    for(size_t task=0; task < helper.size(); task++) {

      size_t ketAlphaSize = helper[task].ketAlphaEnd-helper[task].ketAlphaStart;
      size_t ketBetaSize  = helper[task].ketBetaEnd-helper[task].ketBetaStart;
      
      if( helper.size() != 0 ) {
#pragma omp parallel
        {
	  // round-robin assignment of work to threads
          size_t thread_id = omp_get_thread_num();
          size_t ia_start = thread_id + helper[task].braAlphaStart;
          size_t ia_end   = helper[task].braAlphaEnd;

	  size_t array_size = (2*norb + bit_length - 1 ) / bit_length;
	  std::vector<size_t> DetI(array_size);
	  auto DetJ = DetI;
	  std::vector<int> c(2,0);
	  std::vector<int> d(2,0);
	  
	  if( helper[task].taskType == 2 ) { // beta range are same
	    
	    for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	      for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
		
		size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		                +ib-helper[task].braBetaStart;
		if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
		
		DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetI);
		
		// single alpha excitation
		for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		  size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		  size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                  +ib-helper[task].ketBetaStart;
		  OneDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
				     helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
				     onebody_t[thread_id],twobody_t[thread_id]);
		  /*
		  DetFromAlphaBeta(adet[ja],bdet[ib],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					  bit_length,norb,c,d,
					  onebody_t[thread_id],twobody_t[thread_id]);
		  */
		}
		// double alpha excitation
		for(size_t j=0; j < helper[task].DoublesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		  size_t ja = helper[task].DoublesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		  size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                 + ib-helper[task].ketBetaStart;
		  TwoDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+0],
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+1],
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+2],
				     helper[task].DoublesAlphaCrAnSM[ia-helper[task].braAlphaStart][4*j+3],
				     onebody_t[thread_id],twobody_t[thread_id]);
		  /*
		  DetFromAlphaBeta(adet[ja],bdet[ib],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					  bit_length,norb,c,d,
					  onebody_t[thread_id],twobody_t[thread_id]);
		  */
		}
		
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	    
	  } else if ( helper[task].taskType == 1 ) { // alpha range are same
	    
	    for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	      for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
		
		size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		  +ib-helper[task].braBetaStart;
		if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
		
		DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetI);
		
		// single beta excitation
		for(size_t j=0; j < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		  size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][j];
		  size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		                 + jb-helper[task].ketBetaStart;
		  OneDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+0],
				     helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*j+1],
				     onebody_t[thread_id],twobody_t[thread_id]);
		  /*
		  DetFromAlphaBeta(adet[ia],bdet[jb],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					  bit_length,norb,c,d,
					  onebody_t[thread_id],twobody_t[thread_id]);
		  */
		}
		// double beta excitation
		for(size_t j=0; j < helper[task].DoublesFromBetaLen[ib-helper[task].braBetaStart]; j++) {
		  size_t jb = helper[task].DoublesFromBetaSM[ib-helper[task].braBetaStart][j];
		  size_t ketIdx = (ia-helper[task].ketAlphaStart) * ketBetaSize
		                 + jb-helper[task].ketBetaStart;
		  TwoDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+0],
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+1],
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+2],
				     helper[task].DoublesBetaCrAnSM[ib-helper[task].braBetaStart][4*j+3],
				     onebody_t[thread_id],twobody_t[thread_id]);
		  /*
		  DetFromAlphaBeta(adet[ia],bdet[jb],bit_length,norb,DetJ);
		  CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					bit_length,norb,c,d,
					  onebody_t[thread_id],twobody_t[thread_id]);
		  */
		}
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	    
	    
	  } else {
	    
	    for(size_t ia = ia_start; ia < ia_end; ia+=num_threads) {
	      for(size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
		
		size_t braIdx = (ia-helper[task].braAlphaStart)*braBetaSize
		                +ib-helper[task].braBetaStart;
		if( (braIdx % mpi_size_h) != mpi_rank_h ) continue;
		
		DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,DetI);
		
		// two-particle excitation composed of single alpha and single beta
		for(size_t j=0; j < helper[task].SinglesFromAlphaLen[ia-helper[task].braAlphaStart]; j++) {
		  size_t ja = helper[task].SinglesFromAlphaSM[ia-helper[task].braAlphaStart][j];
		  for(size_t k=0; k < helper[task].SinglesFromBetaLen[ib-helper[task].braBetaStart]; k++) {
		    size_t jb = helper[task].SinglesFromBetaSM[ib-helper[task].braBetaStart][k];
		    size_t ketIdx = (ja-helper[task].ketAlphaStart)*ketBetaSize
		                    +jb-helper[task].ketBetaStart;
		    TwoDiffCorrelation(DetI,W[braIdx],T[ketIdx],bit_length,norb,
				       helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+0],
				       helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+0],
				       helper[task].SinglesAlphaCrAnSM[ia-helper[task].braAlphaStart][2*j+1],
				       helper[task].SinglesBetaCrAnSM[ib-helper[task].braBetaStart][2*k+1],
				       onebody_t[thread_id],twobody_t[thread_id]);
		    /*
		    DetFromAlphaBeta(adet[ja],bdet[jb],bit_length,norb,DetJ);
		    CorrelationTermAddition(DetI,DetJ,W[braIdx],T[ketIdx],
					    bit_length,norb,c,d,
					    onebody_t[thread_id],twobody_t[thread_id]);
		    */
		  }
		}
		
	      } // end for(size_t ib=ib_start; ib < ib_end; ib++)
	    } // end for(size_t ia=helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++)
	  } // if ( helper[task].taskType ==  )

        } // end #pragma omp parallel

      } // end if( helper.size() != 0 )
      
      if( helper[task].taskType == 0 && task != helper.size()-1 ) {
	int adetslide = helper[task].adetShift-helper[task+1].adetShift;
	int bdetslide = helper[task].bdetShift-helper[task+1].bdetShift;
	R.resize(T.size());
	std::memcpy(R.data(),T.data(),T.size()*sizeof(ElemT));
	Mpi2dSlide(R,T,adet_comm_size,bdet_comm_size,adetslide,bdetslide,b_comm);
      }
	
    } // end for(size_t task=0; task < helper.size(); task++)

    for(size_t tid = 0; tid < num_threads; tid++) {
#pragma omp parallel for
      for(size_t i=0; i < norb*norb; i++) {
	for(size_t s=0; s < onebody.size(); s++) {
	  onebody[s][i] += onebody_t[tid][s][i];
	}
      }
    }

    for(size_t tid = 0; tid < num_threads; tid++) {
#pragma omp parallel for
      for(size_t i=0; i < norb*norb*norb*norb; i++) {
	for(size_t s=0; s < twobody.size(); s++) {
	  twobody[s][i] += twobody_t[tid][s][i];
	}
      }
    }
    
    for(int s=0; s < 2; s++) {
      MpiAllreduce(onebody[s],MPI_SUM,b_comm);
      MpiAllreduce(onebody[s],MPI_SUM,t_comm);
      MpiAllreduce(onebody[s],MPI_SUM,h_comm);
    }
    for(int s=0; s < 4; s++) {
      MpiAllreduce(twobody[s],MPI_SUM,b_comm);
      MpiAllreduce(twobody[s],MPI_SUM,t_comm);
      MpiAllreduce(twobody[s],MPI_SUM,h_comm);
    }

  }

  
} // end namespace sbd

#endif // end if for #ifndef SBD_CHEMISTRY_PTMB_CORRELATION_H
