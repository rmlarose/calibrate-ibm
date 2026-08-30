/**
@file sbd/chemistry/basic/qcham.h
@brief Function to make Hamiltonian
*/
#ifndef SBD_CHEMISTRY_BASIC_QCHAM_H
#define SBD_CHEMISTRY_BASIC_QCHAM_H

namespace sbd {

  template <typename ElemT>
  void makeQCham(const std::vector<std::vector<size_t>> & dets,
		 const size_t bit_length,
		 const size_t norbs,
		 const Helpers & helper,
		 ElemT & I0,
		 oneInt<ElemT> & I1,
		 twoInt<ElemT> & I2,
		 const size_t StartIdx,
		 const size_t EndIdx,
		 std::vector<ElemT> & hii,
		 std::vector<std::vector<size_t>> & ij,
		 std::vector<std::vector<ElemT>> & hij,
		 int data_width,
		 MPI_Comm comm) {
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm,&mpi_rank);
    MPI_Comm_size(comm,&mpi_size);
    size_t offset = 0;

    hii.resize(dets.size(),ElemT(0.0));
    // diagonal elements
    for(size_t k=StartIdx; k < EndIdx; k++) {
      if( ( k % mpi_size ) != mpi_rank || k < std::max(StartIdx, offset) ) continue;
      hii[k] = ZeroExcite(dets[k],bit_length,norbs,I0,I1,I2);
    }

    ij.resize(dets.size(),ElemT(0.0));
    hij.resize(dets.size(),ElemT(0.0));

    // alpha-beta excitation
    
    for(size_t i=0; i < helper.AlphaMajorToBetaSM.size(); i++) {
      for(size_t ii=0; ii < helper.AlphaMajorToBetaLen[i]; ii++) {
	size_t Astring = i;
	size_t Bstring = helper.AlphaMajorToBetaSM[i][ii];
	size_t DetI = helper.AlphaMajorToDetSM[i][ii];
	if ( ( (DetI-1) % mpi_size != mpi_rank ) || (DetI-1) < StartIdx ) continue;

	// single alpha excitation
	for(size_t j=0; j < helper.SinglesFromAlphaLen[Astring]; j++) {
	  size_t Asingle = helper.SinglesFromAlphaSM[Astring][j];
	  auto itA = std::lower_bound(&helper.BetaMajorToAlphaSM[Bstring][0],
				      &helper.BetaMajorToAlphaSM[Bstring][0]
				      +helper.BetaMajorToAlphaLen[Bstring],
				      Asingle);
	  if( itA != (&helper.BetaMajorToAlphaSM[Bstring][0]
		      + helper.BetaMajorToAlphaLen[Bstring]) ) {
	    int indexA = std::distance(&helper.BetaMajorToAlphaSM[Bstring][0],itA);
	    int DetJ = helper.BetaMajorToDetSM[Bstring][indexA];
	    if( DetJ >= DetI ) continue;
	    size_t orbDiff;
	    ElemT eij = Hij(dets[DetI-1],dets[DetJ-1],
			    bit_length,norbs,I0,I1,I2,orbDiff);
	    ij[(DetI-1)/mpi_size].push_back(DetJ-1);
	    hij[(DetI-1)/mpi_size].push_back(eij);
	  }
	}

	// two-particle excitaiton composed of single alpha and single beta
	for(size_t j=0; j < helper.SinglesFromAlphaLen[Astring]; j++) {
	  
	  size_t Asingle = helper.SinglesFromAlphaSM[Astring][j];
	  size_t SearchStartIndex = 0;
	  size_t AlphaToBetaLen = helper.AlphaMajorToBetaLen[Asingle];
	  size_t SinglesFromBLen = helper.SinglesFromBetaLen[Bstring];
	  size_t maxAToB =
            helper.AlphaMajorToBetaSM[Asingle][helper.AlphaMajorToBetaLen[Asingle] - 1];
	  for(size_t k=0; k < SinglesFromBLen; k++) {
	    size_t & Bsingle = helper.SinglesFromBetaSM[Bstring][k];
	    if( SearchStartIndex >= AlphaToBetaLen ) break;
	    
	    int index = std::distance(&helper.AlphaMajorToBetaSM[Asingle][0],
			std::lower_bound(&helper.AlphaMajorToBetaSM[Asingle][0]
					 + SearchStartIndex,
					 &helper.AlphaMajorToBetaSM[Asingle][0]
					 + AlphaToBetaLen,
					 Bsingle));
	    SearchStartIndex = index;
	    if (index < AlphaToBetaLen &&
		helper.AlphaMajorToBetaSM[Asingle][index] == Bsingle) {
	      int DetJ = helper.AlphaMajorToDetSM[Asingle][SearchStartIndex];
	      
	      if (DetJ >= DetI) continue;
	      
	      size_t orbDiff;
	      ElemT eij = Hij(dets[DetJ-1], dets[DetI-1],
			      bit_length,norbs,I0,I1,I2,orbDiff);
	      ij[(DetI-offset-1)/mpi_size].push_back(DetJ-1);
	      hij[(DetI-offset-1)/mpi_size].push_back(eij);
	    }
	  }
	}
	
	// single beta excitation
	for (int j = 0; j < helper.SinglesFromBetaLen[Bstring]; j++) {
	  int Bsingle = helper.SinglesFromBetaSM[Bstring][j];

	  // Find Bsingle by bisection
	  auto it = std::lower_bound(&helper.AlphaMajorToBetaSM[Astring][0],
				     &helper.AlphaMajorToBetaSM[Astring][0]
				     + helper.AlphaMajorToBetaLen[Astring],
				     Bsingle);
	  if( it != (&helper.AlphaMajorToBetaSM[Astring][0]+helper.AlphaMajorToBetaLen[Astring]) ) {
	    int index = std::distance(&helper.AlphaMajorToBetaSM[Astring][0], it);
	    if ( helper.AlphaMajorToBetaSM[Astring][index] != Bsingle) {
	      continue; //
	    }
	  
	    int DetJ = helper.AlphaMajorToDetSM[Astring][index];
	  
	    if (DetJ >= DetI) continue;
	    
	    size_t orbDiff;
	    ElemT eij = Hij(dets[DetJ-1],dets[DetI-1],
			    bit_length,norbs,I0,I1,I2,orbDiff);
	    
	    ij[(DetI-offset-1)/mpi_size].push_back(DetJ - 1);
	    hij[(DetI-offset-1)/mpi_size].push_back(eij);
	  }
	}

	// double beta excitation
	for (size_t j = 0; j < helper.AlphaMajorToBetaLen[i]; j++) {
	  int DetJ = helper.AlphaMajorToDetSM[i][j];
	  if (DetJ >= DetI) continue;

	  const std::vector<size_t> & DetAtJ = dets[DetJ - 1];
	  const std::vector<size_t> & DetAtI = dets[DetI - 1];
	  
	  // ExcitationDistance
	  if( difference(DetAtI,DetAtJ,bit_length,2*norbs) == 4 ) {
	    size_t orbDiff;
	    ElemT eij = Hij(DetAtJ,DetAtI,
			    bit_length,norbs,I0,I1,I2,orbDiff);
	    ij[(DetI-offset-1)/mpi_size].push_back(DetJ - 1);
	    hij[(DetI-offset-1)/mpi_size].push_back(eij);
	  }
	}

	// double alpha excitation
	for (int j = 0; j < helper.BetaMajorToAlphaLen[Bstring]; j++) {
	  int DetJ = helper.BetaMajorToDetSM[Bstring][j];
	  
	  if (DetJ >= DetI) continue;
	  
	  const std::vector<size_t> & DetAtJ = dets[DetJ - 1];
	  const std::vector<size_t> & DetAtI = dets[DetI - 1];
	  
	  // ExcitationDistance
	  if ( difference(DetAtI,DetAtJ,bit_length,2*norbs) == 4 ) {
	    size_t orbDiff;
	    ElemT eij = Hij(DetAtJ,DetAtI,
			    bit_length,norbs,I0,I1,I2,orbDiff);
	    ij[(DetI-offset-1)/mpi_size].push_back(DetJ - 1);
	    hij[(DetI-offset-1)/mpi_size].push_back(eij);
	  }
	}
	
      } // end for loop for ii
    } // end for loop for i
    
  }
  
  template <typename ElemT>
  void makeQCham(const std::vector<std::vector<size_t>> & dets,
		 const size_t bit_length,
		 const size_t norbs,
		 const Helpers & helper,
		 ElemT & I0,
		 oneInt<ElemT> & I1,
		 twoInt<ElemT> & I2,
		 const size_t StartIdx,
		 const size_t EndIdx,
		 std::vector<ElemT> & hii,
		 std::vector<std::vector<std::vector<size_t>>> & ih,
		 std::vector<std::vector<std::vector<size_t>>> & jh,
		 std::vector<std::vector<std::vector<size_t>>> & tr,
		 std::vector<std::vector<std::vector<ElemT>>> & hij,
		 int data_width,
		 MPI_Comm h_comm,
		 MPI_Comm b_comm) {
    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    int mpi_rank_b = 0;
    int mpi_size_b = 1;
    MPI_Comm_rank(h_comm,&mpi_rank_h);
    MPI_Comm_size(h_comm,&mpi_size_h);
    MPI_Comm_rank(b_comm,&mpi_rank_b);
    MPI_Comm_size(b_comm,&mpi_size_b);
    size_t offset = 0;
    size_t num_threads = 1;

    hii.resize(dets.size(),ElemT(0.0));
    // diagonal elements
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
#pragma omp for
      for(size_t k=StartIdx; k < EndIdx; k++) {
	if( ( k % mpi_size_h ) != mpi_rank_h || k < std::max(StartIdx, offset) ) continue;
	hii[k] = ZeroExcite(dets[k],bit_length,norbs,I0,I1,I2);
      }
    }

    std::vector<std::vector<std::vector<size_t>>> tdets(data_width);
    int inc_size = (data_width-1)/2;
    int dec_size = (data_width)/2;
    for(int d=0; d < inc_size; d++) {
      if( d == 0 ) {
	MpiSlide(dets,tdets[d],1,b_comm);
      } else {
	MpiSlide(tdets[inc_size-d],tdets[inc_size-d-1],1,b_comm);
      }
    }
    tdets[inc_size] = dets;
    for(int d=0; d < dec_size; d++) {
      if( d == 0 ) {
	MpiSlide(dets,tdets[inc_size-d-1],1,b_comm);
      } else {
	MpiSlide(tdets[inc_size-d],tdets[inc_size-d-1],1,b_comm);
      }
    }
    int mpi_round = mpi_size_b / data_width;

    ih.resize(mpi_round,std::vector<std::vector<size_t>>(num_threads));
    jh.resize(mpi_round,std::vector<std::vector<size_t>>(num_threads));
    tr.resize(mpi_round,std::vector<std::vector<size_t>>(num_threads));
    hij.resize(mpi_round,std::vector<std::vector<ElemT>>(num_threads));

    for(int round=0; round < mpi_round; round++) {

      size_t chunk_size = helper.AlphaMajorToBetaSM.size() / num_threads;
      
#pragma omp parallel
      {

	size_t thread_id = omp_get_thread_num();
	size_t i_start = thread_id * chunk_size;
	size_t i_end = (thread_id+1) * chunk_size;
	if( thread_id == num_threads - 1 ) {
	  i_end = helper.AlphaMajorToBetaSM.size();
	}

	std::vector<size_t> local_ih;
	std::vector<size_t> local_jh;
	std::vector<size_t> local_tr;
	std::vector<ElemT> local_hij;
	
	// alpha-beta excitation
	
	for(size_t i=i_start; i < i_end; i++) {
	  for(size_t ii=0; ii < helper.AlphaMajorToBetaLen[i]; ii++) {
	    
	    size_t Astring = i;
	    size_t Bstring = helper.AlphaMajorToBetaSM[i][ii];
	    size_t DetI = helper.AlphaMajorToDetSM[i][ii];
	    if ( ((DetI-1) % mpi_size_h != mpi_rank_h ) ||
		 ((DetI-1) < StartIdx) ) continue;
	    
	    // single alpha excitation
	    for(size_t j=0; j < helper.SinglesFromAlphaLen[Astring]; j++) {
	      size_t Asingle = helper.SinglesFromAlphaSM[Astring][j];
	      auto itA = std::lower_bound(&helper.BetaMajorToAlphaSM[Bstring][0],
					  &helper.BetaMajorToAlphaSM[Bstring][0]
					  +helper.BetaMajorToAlphaLen[Bstring],
					  Asingle);
	      if( itA != (&helper.BetaMajorToAlphaSM[Bstring][0]
			 + helper.BetaMajorToAlphaLen[Bstring]) ) {
		int indexA = std::distance(&helper.BetaMajorToAlphaSM[Bstring][0],itA);
		if( helper.BetaMajorToAlphaSM[Bstring][indexA] != Asingle ) continue;
		
		int DetJ = helper.BetaMajorToDetSM[Bstring][indexA];
		if(DetJ == DetI) continue;
		size_t orbDiff;
		ElemT eij = Hij(dets[DetI-1],dets[DetJ-1],
				bit_length,norbs,I0,I1,I2,orbDiff);
		if( std::abs(eij) > 1.0e-8 ) {
		  local_ih.push_back(DetI-1);
		  local_jh.push_back(DetJ-1);
		  local_tr.push_back(0);
		  local_hij.push_back(eij);
		}
	      }
	    }
	    
	    // two-particle excitaiton composed of single alpha and single beta
	    for(size_t j=0; j < helper.SinglesFromAlphaLen[Astring]; j++) {
	      
	      size_t Asingle = helper.SinglesFromAlphaSM[Astring][j];
	      size_t SearchStartIndex = 0;
	      size_t AlphaToBetaLen = helper.AlphaMajorToBetaLen[Asingle];
	      size_t SinglesFromBLen = helper.SinglesFromBetaLen[Bstring];
	      for(size_t k=0; k < SinglesFromBLen; k++) {
		size_t & Bsingle = helper.SinglesFromBetaSM[Bstring][k];
		if( SearchStartIndex >= AlphaToBetaLen ) break;
		
		int index = std::distance(&helper.AlphaMajorToBetaSM[Asingle][0],
			    std::lower_bound(&helper.AlphaMajorToBetaSM[Asingle][0]
					     + SearchStartIndex,
					     &helper.AlphaMajorToBetaSM[Asingle][0]
					     + AlphaToBetaLen,
					     Bsingle));
		SearchStartIndex = index;
		if (index < AlphaToBetaLen &&
		    helper.AlphaMajorToBetaSM[Asingle][index] == Bsingle) {
		  int DetJ = helper.AlphaMajorToDetSM[Asingle][SearchStartIndex];
		  
		  if (DetJ == DetI) continue;
		  
		  size_t orbDiff;
		  ElemT eij = Hij(dets[DetJ-1], dets[DetI-1],
				  bit_length,norbs,I0,I1,I2,orbDiff);
		  if( std::abs(eij) > 1.0e-8 ) {
		    local_ih.push_back(DetI-1);
		    local_jh.push_back(DetJ-1);
		    local_tr.push_back(0);
		    local_hij.push_back(eij);
		  }
		}
	      }
	    }
	    
	    // single beta excitation
	    for (int j = 0; j < helper.SinglesFromBetaLen[Bstring]; j++) {
	      int Bsingle = helper.SinglesFromBetaSM[Bstring][j];
	      auto it = std::lower_bound(&helper.AlphaMajorToBetaSM[Astring][0],
					 &helper.AlphaMajorToBetaSM[Astring][0]
					 + helper.AlphaMajorToBetaLen[Astring],
					 Bsingle);
	      if( it != (&helper.AlphaMajorToBetaSM[Astring][0]
			 +helper.AlphaMajorToBetaLen[Astring]) ) {
		int index = std::distance(&helper.AlphaMajorToBetaSM[Astring][0], it);
		if ( helper.AlphaMajorToBetaSM[Astring][index] != Bsingle ) {
		  continue; //
		}
	      
		int DetJ = helper.AlphaMajorToDetSM[Astring][index];
	      
		if (DetJ == DetI) continue;
		
		size_t orbDiff;
		ElemT eij = Hij(dets[DetJ-1],dets[DetI-1],
				bit_length,norbs,I0,I1,I2,orbDiff);
		if( std::abs(eij) > 1.0e-8 ) {
		  local_ih.push_back(DetI-1);
		  local_jh.push_back(DetJ-1);
		  local_tr.push_back(0);
		  local_hij.push_back(eij);
		}
	      }
	    }
	    
	    // double beta excitation
	    for (size_t j = 0; j < helper.AlphaMajorToBetaLen[i]; j++) {
	      int DetJ = helper.AlphaMajorToDetSM[i][j];
	      if (DetJ == DetI) continue;
	      
	      const std::vector<size_t> & DetAtJ = dets[DetJ - 1];
	      const std::vector<size_t> & DetAtI = dets[DetI - 1];
	      
	      // ExcitationDistance
	      if( difference(DetAtI,DetAtJ,bit_length,2*norbs) == 4 ) {

		size_t orbDiff;
		ElemT eij = Hij(DetAtJ,DetAtI,
				bit_length,norbs,I0,I1,I2,orbDiff);
		if( std::abs(eij) > 1.0e-8 ) {
		  local_ih.push_back(DetI-1);
		  local_jh.push_back(DetJ-1);
		  local_tr.push_back(0);
		  local_hij.push_back(eij);
		}
	      }
	    }
	    
	    // double alpha excitation
	    for (int j = 0; j < helper.BetaMajorToAlphaLen[Bstring]; j++) {
	      int DetJ = helper.BetaMajorToDetSM[Bstring][j];
	      
	      if (DetJ == DetI) continue;
	      
	      const std::vector<size_t> & DetAtJ = dets[DetJ - 1];
	      const std::vector<size_t> & DetAtI = dets[DetI - 1];
	      
	      // ExcitationDistance
	      if ( difference(DetAtI,DetAtJ,bit_length,2*norbs) == 4 ) {

		size_t orbDiff;
		ElemT eij = Hij(DetAtJ,DetAtI,
				bit_length,norbs,I0,I1,I2,orbDiff);
		if( std::abs(eij) > 1.0e-8 ) {
		  local_ih.push_back(DetI-1);
		  local_jh.push_back(DetJ-1);
		  local_tr.push_back(0);
		  local_hij.push_back(eij);
		}
	      }
	    }
	  } // end for loop for ii
	} // end for loop for i

#pragma omp critical
	{
	  ih[round][thread_id].insert(ih[round][thread_id].end(),
				      std::make_move_iterator(local_ih.begin()),
				      std::make_move_iterator(local_ih.end()));
	  jh[round][thread_id].insert(jh[round][thread_id].end(),
				      std::make_move_iterator(local_jh.begin()),
				      std::make_move_iterator(local_jh.end()));
	  tr[round][thread_id].insert(tr[round][thread_id].end(),
				      std::make_move_iterator(local_tr.begin()),
				      std::make_move_iterator(local_tr.end()));
	  hij[round][thread_id].insert(hij[round][thread_id].end(),
				       std::make_move_iterator(local_hij.begin()),
				       std::make_move_iterator(local_hij.end()));
	}
      } // end pragma omp parallel

      if( mpi_round != 1 ) {
	std::vector<std::vector<size_t>> temp;
	for(size_t i=0; i < tdets.size(); i++) {
	  MpiSlide(tdets[i],temp,data_width,b_comm);
	  tdets[i] = temp;
	}
      }
      
    } // end for loop for mpi_round
      
  }
  

  
} // end namespace sbd

#endif // SBD_CHEMISTRY_QCHAM_H

