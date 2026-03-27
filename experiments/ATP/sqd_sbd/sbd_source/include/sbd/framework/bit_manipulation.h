/// This is a part of sbd
/**
@file bit_manipulation.h
@brief mathematical tools for bit manipulations
*/

#ifndef SBD_FRAMEWORK_BIT_MANIPULATION_H
#define SBD_FRAMEWORK_BIT_MANIPULATION_H

#include <stdint.h>
#include <limits.h>
#include <unistd.h>
#include <stdexcept>
#include <bitset>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <fstream>


#include "mpi.h"

#define SBD_BIT_LENGTH 20

  bool operator < (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();

    assert( a_size == b_size );

    bool res = false;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] < b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] > b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator <= (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size < b_size ) {
      return true;
    } else if ( a_size > b_size ) {
      return false;
    }

    bool res = true;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] < b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] > b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator > (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size > b_size ) {
      return true;
    } else if ( a_size < b_size ) {
      return false;
    }

    bool res = false;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] > b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] < b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator >= (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size > b_size ) {
      return true;
    } else if ( a_size < b_size ) {
      return false;
    }

    bool res = true;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] > b[n-1] ) {
	res = true;
	break;
      } else if ( a[n-1] < b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

  bool operator == (const std::vector<size_t> & a, const std::vector<size_t> & b) {
    size_t a_size = a.size();
    size_t b_size = b.size();
    if( a_size != b_size ) {
      return false;
    }
    bool res = true;
    for(size_t n = a_size; n > 0; n--) {
      if( a[n-1] != b[n-1] ) {
	res = false;
	break;
      }
    }
    return res;
  }

namespace sbd {

  /**
     Function for making a string from the bitarray data
     @param[in] config: target bitarray
     @param[in] bit_length: length of the bitstring managed by each size_t
     @param[in] L: number of total bits in the bitstring
   */
  std::string makestring(const std::vector<size_t> & config,
			 size_t bit_length,
			 size_t L) {
    std::string s;
    for(size_t i=L; i > 0; i--) {
      size_t p = (i-1) % bit_length;
      size_t b = (i-1) / bit_length;
      if( config[b] & (static_cast<size_t>(1) << p) ) {
	s += std::string("1").c_str();
      }
      else {
	s += std::string("0").c_str();
      }
    }
    return s;
  }

  /**
     Function for construct the bitarray data from a string
     @param[in] s: target string
     @param[in] bit_length: length of the bitstring managed by each size_t
     @param[in] L: number of total bits in the bitstring
   */
  std::vector<size_t> from_string(const std::string & s,
				  size_t bit_length,
				  size_t L) {
    size_t num_words = (L + bit_length - 1) / bit_length;
    std::vector<size_t> result(num_words, 0);
    for (size_t i = 0; i < L; ++i) {
      char bit = s[L - 1 - i];
      if (bit == '1') {
	size_t p = i % bit_length;
	size_t b = i / bit_length;
	result[b] |= (static_cast<size_t>(1) << p);
      }
    }
    return result;
  }

  /**
     Function for finding a mpi process which manages the target bit string
     @param[in] config: target configuration
     @param[in] config_begin: first element for each mpi
  */
  void mpi_process_search(const std::vector<size_t> & target_config,
			  const std::vector<std::vector<size_t>> & config_begin,
			  const std::vector<std::vector<size_t>> & config_end,
			  int & target_mpi_rank, bool & mpi_exist) {
    size_t mpi_size = config_begin.size();
    mpi_exist = false;
    for(int rank=0; rank < mpi_size; rank++) {
      if( config_begin[rank] <= target_config && target_config < config_end[rank] ) {
	target_mpi_rank = rank;
	mpi_exist = true;
	break;
      }
    }
  }

  /**
     Function for finding the state index of target bit string
     @param[in] target_config: target configuration
     @param[in] config: all configuration managed by target mpi process
     @param[in] index_begin: index of first element managed by target mpi process
     @param[in] index_end: +1 index of last element managed by target mpi process
  */

  void bisection_search(const std::vector<size_t> & target_config,
			const std::vector<std::vector<size_t>> & config,
			const size_t & index_begin,
			const size_t & index_end,
			size_t & index, bool & exist) {
    bool do_bisection = true;
    size_t index_a = index_begin;
    size_t index_b = index_end-1;
    exist = false;
    while( do_bisection ) {
      size_t index_c = (index_a + index_b)/2;
      if( config[index_c] < target_config ) {
	index_a = index_c+1;
      } else if ( config[index_c] > target_config ) {
	index_b = index_c-1;
      } else {
	index = index_c;
	do_bisection = false;
	exist = true;
	return;
      }
      if( (index_b - index_a) < 2 && do_bisection ) {
	do_bisection = false;
	if( config[index_b] == target_config ) {
	  index = index_b;
	  exist = true;
	} else if ( config[index_a] == target_config ) {
	  index = index_a;
	  exist = true;
	} else if ( target_config < config[index_a] ) {
	  index = index_a;
	  exist = false;
	} else if( target_config < config[index_b] ) {
	  index = index_b;
	  exist = false;
	} else {
	  index = index_b+1;
	  exist = false;
	}
      }
    }
  }

  /**
     Function for finding the state index of target bit string
     @param[in] target_config: target configuration
     @param[in] config: all configuration managed by target mpi process
     @param[in] index_begin: index of first element managed by target mpi process
     @param[in] index_end: +1 index of last element managed by target mpi process
  */
  void bisection_search_mpi(const std::vector<size_t> & target_config,
			    const std::vector<std::vector<size_t>> & config,
			    const size_t & index_begin,
			    const size_t & index_end,
			    int target_mpi_rank, size_t & index, bool & exist,
			    MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    if( mpi_rank == target_mpi_rank ) {
      bool do_bisection = true;
      std::vector<size_t> config_a = config[index_begin];
      std::vector<size_t> config_b = config[index_end-1];
      size_t index_a = index_begin;
      size_t index_b = index_end-1;
      while( do_bisection ) {
	size_t index_c = (index_a + index_b)/2;
	std::vector<size_t> config_c = config[index_c-index_begin];
	if( config_c < target_config ) {
	  config_a = config_c;
	  index_a = index_c;
	} else if ( config_c > target_config ) {
	  config_b = config_c;
	  index_b = index_c;
	} else {
	  index = index_c;
	  do_bisection = false;
	  exist = true;
	}
	if( index_b-index_a < 2 ) {
	  do_bisection = false;
	  if( config_b == target_config ) {
	    index = index_b;
	    exist = true;
	  } else if ( config_a == target_config ) {
	    index = index_a;
	    exist = true;
	  } else if ( target_config < config_a ) {
	    index = index_a;
	    exist = false;
	  } else if( target_config < config_b ) {
	    index = index_b;
	    exist = false;
	  } else {
	    index = index_b+1;
	    exist = false;
	  }
	}
      }
    } // end if for mpi process
  } // end void function bisection_search

  /**
     Function for adding 1 to the bitstring
     @param[in/out] a: bitstring
     @param[in] bit_length: length for the bitstring managed by each size_t
   */

  void bitadvance(std::vector<size_t> & a, int bit_length) {
    size_t x;
    size_t d = (((size_t) 1) << bit_length) - 1;
    size_t v = (size_t) 1;
    for(size_t n=0; n < a.size(); n++) {
      x = a[n]+v;
      a[n] = (x & d);
      v = x >> bit_length;
    }
  }

  /**
     Function for sorting the array of bit strings.
     @param[in/out] a: set of bit strings to be sorted
   */
  void sort_bitarray(std::vector<std::vector<size_t>> & a) {
    std::sort(a.begin(),a.end(),
	      [](const std::vector<size_t> & x,
		 const std::vector<size_t> & y)
	      { return x < y; });
    a.erase(std::unique(a.begin(),a.end()),a.end());
  }

  /**
     Function to redistribute the bit string on each mpi processes.
     @param[in/out] config: set of bit strings to be redistributed
     @param[in/out] config_begin: first bit string for each mpi process.
     @param[in/out] config_end: last bit string for each mpi process.
     @param[in/out] index_begin: first index of bit string for each mpi process.
     @param[in/out] index_end: last index of bit string for each mpi process.
     @param[in] total_bit_length: total bit length represented by `std::vector<size_t>`
     @param[in] bit_length: length of bit string managed by each `size_t`
     @param[in] comm: mpi communicator
   */
  void mpi_redistribution(std::vector<std::vector<size_t>> & config,
			  std::vector<std::vector<size_t>> & config_begin,
			  std::vector<std::vector<size_t>> & config_end,
			  std::vector<size_t> & index_begin,
			  std::vector<size_t> & index_end,
			  size_t total_bit_length,
			  size_t bit_length,
			  MPI_Comm comm) {

    int mpi_master = 0;
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    size_t config_length = (total_bit_length+bit_length-1) / bit_length;
    std::vector<size_t> send_config_size(mpi_size,0);
    std::vector<size_t> config_size(mpi_size,0);
    send_config_size[mpi_rank] = config.size();
    MPI_Allreduce(send_config_size.data(),config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
    size_t total_size = 0;
    for(int rank=0; rank < mpi_size; rank++) {
      total_size += config_size[rank];
    }
    std::fill(index_begin.begin(),index_begin.end(),static_cast<size_t>(0));
    std::fill(index_end.begin(),index_end.end(),static_cast<size_t>(0));
    for(int rank=1; rank < mpi_size; rank++) {
      index_begin[rank] = index_begin[rank-1] + config_size[rank-1];
    }
    for(int rank=0; rank < mpi_size; rank++) {
      index_end[rank] = index_begin[rank] + config_size[rank];
    }
    std::vector<size_t> i_begin(mpi_size,0);
    std::vector<size_t> i_end(mpi_size,total_size);
    for(size_t rank=0; rank < mpi_size; rank++) {
      get_mpi_range(mpi_size,rank,i_begin[rank],i_end[rank]);
    }

    size_t new_config_size = i_end[mpi_rank]-i_begin[mpi_rank];
    std::vector<std::vector<size_t>> new_config;
    for(int recv_rank=0; recv_rank < mpi_size; recv_rank++) {
      // find i_begin and i_end mpi process
      int mpi_rank_begin = 0;
      int mpi_rank_end   = mpi_size-1;
      for(int rank=0; rank < mpi_size; rank++) {
	if ( ( index_begin[rank] <= i_begin[recv_rank] )
	     && ( i_begin[recv_rank] < index_end[rank] ) ) {
	  mpi_rank_begin = rank;
	  break;
	}
      }
      for(int rank=mpi_size-1; rank > -1; rank--) {
	if ( ( index_begin[rank] < i_end[recv_rank] ) && ( i_end[recv_rank] <= index_end[rank] ) ) {
	  mpi_rank_end = rank;
	  break;
	}
      }
      for(int send_rank=mpi_rank_begin; send_rank <= mpi_rank_end; send_rank++) {
	if( mpi_rank == send_rank ) {
	  size_t ii_min = std::max(i_begin[recv_rank],index_begin[send_rank]);
	  size_t ii_max = std::min(i_end[recv_rank],  index_end[send_rank]);
	  std::vector<std::vector<size_t>> config_transfer;
	  if( (ii_max - ii_min) > 0 ) {
	    config_transfer.resize(ii_max-ii_min);
	    for(size_t i=ii_min; i < ii_max; i++) {
	      config_transfer[i-ii_min] = config[i-index_begin[send_rank]];
	    }
	  } else {
	    config_transfer.resize(0);
	  }
	  if( send_rank != recv_rank ) {
	    MpiSend(config_transfer,recv_rank,comm);
	  } else {
	    new_config.insert(new_config.end(),config_transfer.begin(),config_transfer.end());
	  }
	}
	if( ( mpi_rank == recv_rank ) && ( send_rank != recv_rank ) ) {
	  std::vector<std::vector<size_t>> config_transfer;
	  MpiRecv(config_transfer,send_rank,comm);
	  new_config.insert(new_config.end(),config_transfer.begin(),config_transfer.end());
	}
      } // end for(int send_rank=mpi_rank_begin; send_rank <= mpi_rank_end; send_rank++)
      MPI_Barrier(comm);
    } // end for(int recv_rank=0; recv_rank < mpi_size; recv_rank++)

    sort_bitarray(new_config);
    config = new_config;

    std::fill(send_config_size.begin(),send_config_size.end(),static_cast<size_t>(0));
    std::fill(config_size.begin(),config_size.end(),static_cast<size_t>(0));
    send_config_size[mpi_rank] = config.size();
    MPI_Allreduce(send_config_size.data(),config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
    index_begin[0] = 0;
    for(int rank=1; rank < mpi_size; rank++) {
      index_begin[rank] = index_begin[rank-1] + config_size[rank-1];
    }
    for(int rank=0; rank < mpi_size; rank++) {
      index_end[rank] = index_begin[rank] + config_size[rank];
    }

    for(int rank=0; rank < mpi_size; rank++) {
      if( rank == mpi_rank ) {
	config_begin[rank] = config[0];
      } else {
	config_begin[rank].resize(config_length);
      }

      MPI_Bcast(config_begin[rank].data(),config_length,SBD_MPI_SIZE_T,rank,comm);
    }
    for(int rank=0; rank < mpi_size-1; rank++) {
      config_end[rank] = config_begin[rank+1];
    }
    if( mpi_rank == mpi_size-1 ) {
      config_end[mpi_rank] = config[config.size()-1];
      bitadvance(config_end[mpi_rank],bit_length);
    } else {
      config_end[mpi_size-1].resize(config_length);
    }
    MPI_Bcast(config_end[mpi_size-1].data(),config_length,SBD_MPI_SIZE_T,mpi_size-1,comm);

  }

  /**
     Function to sort the set of bit string distributed on the all mpi processes.
     @param[in/out] config: set of bit strings to be redistributed
     @param[in/out] config_begin: first bit string for each mpi process.
     @param[in/out] config_end: last bit string for each mpi process.
     @param[in/out] index_begin: first index of bit string for each mpi process.
     @param[in/out] index_end: last index of bit string for each mpi process.
     @param[in] total_bit_length: total bit length represented by `std::vector<size_t>`
     @param[in] bit_length: length of bit string managed by each `size_t`
     @param[in] comm: mpi communicator
   */
  void mpi_sort_bitarray(std::vector<std::vector<size_t>> & config,
			 std::vector<std::vector<size_t>> & config_begin,
			 std::vector<std::vector<size_t>> & config_end,
			 std::vector<size_t> & index_begin,
			 std::vector<size_t> & index_end,
			 size_t total_bit_length,
			 size_t bit_length,
			 MPI_Comm comm,
			 int id=0) {
    int mpi_master = 0;
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    size_t bit_size = (total_bit_length + bit_length - 1 ) / bit_length;
    if( mpi_size == 1 ) {
      sort_bitarray(config);
      config_begin[mpi_rank] = config[0];
      config_end[mpi_rank] = config[config.size()-1];
      index_begin[mpi_rank] = 0;
      index_end[mpi_rank] = config.size()-1;
    } else {
      // mpi_size = 2 -> mpi_size_half = 1, mpi_rank / mpi_size_half = 0 or 1
      // mpi_size = 3 -> mpi_size_half = 2, mpi_rank / mpi_size_half = 0 or 1
      int mpi_ierr;
      int mpi_size_half = (mpi_size / 2) + ( mpi_size % 2 );
      int mpi_color = mpi_rank / mpi_size_half;
      int mpi_key   = ( mpi_rank < mpi_size_half ) ? ( mpi_rank ) : ( mpi_rank - mpi_size_half );
      // int mpi_key mpi_rank % mpi_size_half;
      MPI_Comm new_comm;
      int new_id = id + mpi_color * mpi_size_half;
      MPI_Comm_split(comm,mpi_color,mpi_key,&new_comm);
      mpi_sort_bitarray(config,config_begin,config_end,index_begin,index_end,total_bit_length,bit_length,new_comm,new_id);

#ifdef SBD_DEBUG_BIT
      // for(int rank=0; rank < mpi_size; rank++) {
      // if( mpi_rank == rank ) {
      if( mpi_rank == 0 ) {
	std::cout << " ParallelSort at rank " << mpi_rank << " id " << id
		  << ": start parallel merge sort for size " << mpi_size
		  << " (" << mpi_size_half << "," << mpi_size-mpi_size_half << ")" << std::endl;
      }
      // MPI_Barrier(comm);
      // }
      sleep(1);
#endif

      // determine the range to be managed by each node
      int mpi_size_a = mpi_size_half;
      int mpi_size_b = mpi_size - mpi_size_half;
      int mpi_master_a = 0;
      int mpi_master_b = mpi_size_half;
      std::vector<std::vector<size_t>> config_begin_a(mpi_size_a,std::vector<size_t>(bit_size,0));
      std::vector<std::vector<size_t>> config_middle_a(mpi_size_a,std::vector<size_t>(bit_size,0));
      std::vector<std::vector<size_t>> config_end_a(mpi_size_a,std::vector<size_t>(bit_size,0));
      std::vector<std::vector<size_t>> config_begin_b(mpi_size_b,std::vector<size_t>(bit_size));
      std::vector<std::vector<size_t>> config_end_b(mpi_size_b,std::vector<size_t>(bit_size));
      std::vector<size_t> config_end_a_end(bit_size,0);
      std::vector<size_t> config_end_b_end(bit_size,0);
      if( mpi_color == 0 ) {
	config_begin_a[mpi_key] = config[0];
	config_middle_a[mpi_key] = config[config.size()/2];
	if( mpi_key == mpi_size_a - 1 ) {
	  config_end_a_end = config[config.size()-1];
	  bitadvance(config_end_a_end,bit_length);
	}
      }
      if( mpi_color == 1 ) {
	config_begin_b[mpi_key] = config[0];
	if( mpi_key == mpi_size_b - 1 ) {
	  config_end_b_end = config[config.size()-1];
	  bitadvance(config_end_b_end,bit_length);
	}
      }
      for(int rank=0; rank < mpi_size_a; rank++) {
	MPI_Bcast(config_begin_a[rank].data(),bit_size,SBD_MPI_SIZE_T,rank,comm);
	MPI_Bcast(config_middle_a[rank].data(),bit_size,SBD_MPI_SIZE_T,rank,comm);
      }
      for(int rank=0; rank < mpi_size_b; rank++) {
	MPI_Bcast(config_begin_b[rank].data(),bit_size,SBD_MPI_SIZE_T,mpi_master_b+rank,comm);
      }
      MPI_Bcast(config_end_a_end.data(),bit_size,SBD_MPI_SIZE_T,mpi_master_b-1,comm);
      MPI_Bcast(config_end_b_end.data(),bit_size,SBD_MPI_SIZE_T,mpi_size-1,comm);

      for(int rank=0; rank < mpi_size_a-1; rank++) {
	config_end_a[rank] = config_begin_a[rank+1];
      }
      config_end_a[mpi_size_a-1] = config_end_a_end;
      for(int rank=0; rank < mpi_size_b-1; rank++) {
	config_end_b[rank] = config_begin_b[rank+1];
      }
      config_end_b[mpi_size_b-1] = config_end_b_end;

      if( config_end_a_end > config_begin_b[0] ) {

#ifdef SBD_DEBUG_BIT
	MPI_Barrier(comm);
	// for(int rank=0; rank < mpi_size; rank++) {
	// if( mpi_rank == rank ) {
	if( mpi_rank == 0 ) {
	  std::cout << " ParallelSort at rank " << mpi_rank << " id " << id
		      << ": end config_begin and config_end set up, and enter case necessary to merge " << std::endl;
	}
	// MPI_Barrier(comm);
	// }
	sleep(1);
#endif

	for(int rank=0; rank < mpi_size_half; rank++) {
	  if( 2*rank == mpi_size-1 ) {
	    config_begin[2*rank] = config_begin_a[rank];
	    config_end[2*rank] = config_end_a[rank];
	  } else if( 2*rank < mpi_size ) {
	    config_begin[2*rank] = config_begin_a[rank];
	    config_end[2*rank]   = config_middle_a[rank];
	  }
	  if( 2*rank+1 < mpi_size ) {
	    config_begin[2*rank+1] = config_middle_a[rank];
	    config_end[2*rank+1] = config_end_a[rank];
	  }
	}
	if( config_begin_b[0] < config_begin[0] ) {
	  config_begin[0] = config_begin_b[0];
	}
	if( config_end[mpi_size-1] < config_end_b_end ) {
	  config_end[mpi_size-1] = config_end_b_end;
	}

#ifdef SBD_DEBUG_BIT
	MPI_Barrier(comm);
	// for(int rank=0; rank < mpi_size; rank++) {
	//  if( mpi_rank == rank ) {
	if( mpi_rank == 0 ) {
	  std::cout << " ParallelSort at rank " << mpi_rank << " id " << id
		    << ": complete range definition:";
	  for(int rank_t=0; rank_t < mpi_size; rank_t++) {
	    std::cout << " [" << makestring(config_begin[rank_t],bit_length,total_bit_length)
		      << ","  << makestring(config_end[rank_t],bit_length,total_bit_length) << ")";
	  }
	  std::cout << std::endl;
	}
	// MPI_Barrier(comm);
	// }
	sleep(1);
#endif

	std::vector<std::vector<size_t>> new_config_b;
	for(int r_rank=0; r_rank < mpi_size; r_rank++) {
	  for(int s_rank=0; s_rank < mpi_size_b; s_rank++) {

	    if( ( config_begin[r_rank] < config_end_b[s_rank] )
	     && ( config_begin_b[s_rank] < config_end[r_rank] ) ) {

	      if( (s_rank + mpi_master_b) == mpi_rank ) {

		size_t i_begin = 0;
		size_t i_end   = config.size();
		if( config.size() > 0 ) {
		  if( config[0] < config_begin[r_rank] ) {
		    auto itb = std::lower_bound(config.begin(),config.end(),
						config_begin[r_rank],
						[](const std::vector<size_t> & lhs,
						   const std::vector<size_t> & rhs) {
						  return lhs < rhs;
						});
		    i_begin = static_cast<size_t>(std::distance(config.begin(),itb));
		  }
		  if( config_end[r_rank] <= config[config.size()-1] ) {
		    auto ite = std::lower_bound(config.begin(),config.end(),
						config_end[r_rank],
						[](const std::vector<size_t> & lhs,
						   const std::vector<size_t> & rhs) {
						  return lhs < rhs;
						});
		    i_end = static_cast<size_t>(std::distance(config.begin(),ite));
		  }
		}

#ifdef SBD_DEBUG_BIT
		std::cout << " rank = " << mpi_rank << " id " << id << ": send data [" << i_begin << "," << i_end
			  << ") from " << s_rank + mpi_master_b << " to " << r_rank
			  << " because [" << makestring(config_begin[r_rank],bit_length,total_bit_length)
			  << " < " << makestring(config_end_b[s_rank],bit_length,total_bit_length)
			  << "] = " << ( config_begin[r_rank] < config_end_b[s_rank] )
			  << " and [" << makestring(config_begin_b[s_rank],bit_length,total_bit_length)
			  << " < " << makestring(config_end[r_rank],bit_length,total_bit_length)
			  << "] = " << ( config_begin_b[s_rank] < config_end[r_rank] ) << std::endl;
#endif

		std::vector<std::vector<size_t>> config_transfer;
		size_t transfer_size = i_end-i_begin;
		if( i_end-i_begin > 0 ) {
		  config_transfer.resize(transfer_size);
		}
		for(size_t i = i_begin; i < i_end; i++) {
		  config_transfer[i-i_begin] = config[i];
		}
		if( s_rank+mpi_master_b != r_rank ) {
		  MpiSend(config_transfer,r_rank,comm);
		} else {
		  new_config_b.insert(new_config_b.end(),config_transfer.begin(),config_transfer.end());
		}

	      }
	      if( ( r_rank == mpi_rank ) && ( s_rank+mpi_master_b != r_rank ) ) {

		std::vector<std::vector<size_t>> config_transfer(0);
		MpiRecv(config_transfer,s_rank+mpi_master_b,comm);
		new_config_b.insert(new_config_b.end(),config_transfer.begin(),config_transfer.end());

	      }
	    }
	  }
	} // end send configs from b-block

	sort_bitarray(new_config_b);

	std::vector<std::vector<size_t>> new_config_a;
	for(int s_rank=0; s_rank < mpi_size_a; s_rank++) {
	  int r_rank = 2 * s_rank;
	  if( r_rank < mpi_size ) {
	    if( mpi_rank == s_rank ) {
	      size_t i_begin = 0;
	      size_t i_end = config.size();
	      if( r_rank == mpi_size-1 ) {
		i_begin = 0;
		i_end = config.size();
	      } else {
		i_begin = 0;
		i_end = config.size()/2;
	      }
	      std::vector<std::vector<size_t>> config_transfer;
	      config_transfer.resize(0);
	      size_t transfer_size = i_end-i_begin;
	      if( transfer_size > 0 ) {
		config_transfer.resize(transfer_size);
	      }
	      for(size_t i=i_begin; i < i_end; i++) {
		config_transfer[i-i_begin] = config[i];
	      }
	      if( s_rank != r_rank ) {
		MpiSend(config_transfer,r_rank,comm);
	      } else {
		new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	      }
	    }
	    if( mpi_rank == r_rank && s_rank != r_rank ) {
	      std::vector<std::vector<size_t>> config_transfer(0);
	      MpiRecv(config_transfer,s_rank,comm);
	      new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	    }
	  }
	  r_rank = 2*s_rank+1;
	  if( r_rank < mpi_size ) {
	    if( mpi_rank == s_rank ) {
	      size_t i_begin = config.size()/2;
	      size_t i_end = config.size();
	      std::vector<std::vector<size_t>> config_transfer;
	      config_transfer.resize(0);
	      size_t transfer_size = i_end-i_begin;
	      if( transfer_size > 0 ) {
		config_transfer.resize(transfer_size);
	      }
	      for(size_t i=i_begin; i < i_end; i++) {
		config_transfer[i-i_begin] = config[i];
	      }
	      if( s_rank != r_rank ) {
		MpiSend(config_transfer,r_rank,comm);
	      } else {
		new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	      }
	    }
	    if( ( mpi_rank == r_rank ) && ( s_rank != r_rank ) ) {
	      std::vector<std::vector<size_t>> config_transfer(0);
	      MpiRecv(config_transfer,s_rank,comm);
	      new_config_a.insert(new_config_a.end(),config_transfer.begin(),config_transfer.end());
	    }
	  }
	} // end send configs from a-block

	sort_bitarray(new_config_a);

	// Sorted config from sorted a-configs and sorted b-configs (O(N) method)
	size_t config_size = new_config_a.size() + new_config_b.size();
	config.resize(0);
	config.reserve(config_size);

	std::set_union(new_config_a.begin(),new_config_a.end(),
		       new_config_b.begin(),new_config_b.end(),
		       std::back_inserter(config),
		       [](const std::vector<size_t> & lhs,
			  const std::vector<size_t> & rhs) {
			 return lhs < rhs;
		       });

	std::vector<size_t> new_config_size(mpi_size,0);
	std::vector<size_t> send_new_config_size(mpi_size,0);
	send_new_config_size[mpi_rank] = config.size();
	MPI_Allreduce(send_new_config_size.data(),new_config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
	index_begin[0] = 0;
	for(int rank=1; rank < mpi_size; rank++) {
	  index_begin[rank] = index_begin[rank-1] + new_config_size[rank-1];
	}
	for(int rank=0; rank < mpi_size; rank++) {
	  index_end[rank] = index_begin[rank]+new_config_size[rank];
	}


#ifdef SBD_DEBUG_BIT
	if( mpi_rank == 0 ) {
	  std::cout << " [config_begin, config_end) at rank " << mpi_rank << " id " << id << " before redistribution";
	  for(int p_rank=0; p_rank < mpi_size; p_rank++) {
	    std::cout << " [" << makestring(config_begin[p_rank],bit_length,total_bit_length)
		      << "_" << index_begin[p_rank]
		      << "," << makestring(config_end[p_rank],bit_length,total_bit_length)
		      << "_" << index_end[p_rank] << ")";
	  }
	  std::cout << std::endl;
	}
	MPI_Barrier(comm);
	usleep(500000);
	for(int rank=0; rank < mpi_size; rank++) {
	  if( mpi_rank == rank ) {
	    std::cout << " config at rank " << mpi_rank << " id " << id << " before redistribution = [";
	    for(size_t k=0; k < config.size(); k++) {
	      std::cout << ( (k==0) ? "" : "," ) << makestring(config[k],bit_length,total_bit_length);
	    }
	    std::cout << "]" << std::endl;
	  }
	  MPI_Barrier(comm);
	  usleep(100000);
	}
#endif

	mpi_redistribution(config,config_begin,config_end,index_begin,index_end,total_bit_length,bit_length,comm);

#ifdef SBD_DEBUG_BIT
	if( mpi_rank == 0 ) {
	  std::cout << " [config_begin, config_end) at rank " << mpi_rank << " id " << id << " after redistribution";
	  for(int p_rank=0; p_rank < mpi_size; p_rank++) {
	    std::cout << " [" << makestring(config_begin[p_rank],bit_length,total_bit_length)
		      << "_" << index_begin[p_rank]
		      << "," << makestring(config_end[p_rank],bit_length,total_bit_length)
		      << "_" << index_end[p_rank] << ")";
	  }
	  std::cout << std::endl;
	}
	MPI_Barrier(comm);
	usleep(500000);
	for(int rank=0; rank < mpi_size; rank++) {
	  if( mpi_rank == rank ) {
	    std::cout << " config at rank " << mpi_rank << " id " << id << " before redistribution = [";
	    for(size_t k=0; k < config.size(); k++) {
	      std::cout << ( (k==0) ? "" : "," ) << makestring(config[k],bit_length,total_bit_length);
	    }
	    std::cout << "]" << std::endl;
	  }
	  MPI_Barrier(comm);
	  usleep(100000);
	}
#endif


      } else {

	for(int rank=0; rank < mpi_size_a; rank++) {
	  config_begin[rank] = config_begin_a[rank];
	  config_end[rank] = config_end_a[rank];
	}
	for(int rank=0; rank < mpi_size_b; rank++) {
	  config_begin[rank+mpi_master_b] = config_begin_b[rank];
	  config_end[rank+mpi_master_b] = config_end_b[rank];
	}

	std::vector<size_t> new_config_size(mpi_size,0);
	std::vector<size_t> send_new_config_size(mpi_size,0);
	send_new_config_size[mpi_rank] = config.size();
	MPI_Allreduce(send_new_config_size.data(),new_config_size.data(),mpi_size,SBD_MPI_SIZE_T,MPI_SUM,comm);
	index_begin[0] = 0;
	for(int rank=1; rank < mpi_size; rank++) {
	  index_begin[rank] = index_begin[rank-1] + new_config_size[rank-1];
	}
	for(int rank=0; rank < mpi_size; rank++) {
	  index_end[rank] = index_begin[rank]+new_config_size[rank];
	}

	mpi_redistribution(config,config_begin,config_end,index_begin,index_end,total_bit_length,bit_length,comm);

      } // if( config_end_a_end > config_begin_b_begin ) to skip case where it is already sorted.
    }
  }


  /**
     Function to change the length of bit string managed by each `size_t`
     @param[in] bit_length_a: the input length of bit string managed by each `size_t`
     @param[in/out] b: target bit string
     @param[in] bit_length_b: target length of bit string managed by each `size_t`
   */
  void change_bitlength(size_t bit_length_a, std::vector<size_t> & b, size_t bit_length_b) {
    std::vector<size_t> a = b;
    size_t a_size = a.size();
    size_t total_bit_length_a = a_size * bit_length_a;
    size_t b_size = total_bit_length_a / bit_length_b;
    if( total_bit_length_a % bit_length_b != 0 ) {
      b_size++;
    }
    b.resize(b_size);

    size_t a_max_order = bit_length_a;
    size_t min_order = 0;
    size_t b_max_order = bit_length_b;
    size_t maxbit_b = (((size_t) 1) << bit_length_b) - 1;

    size_t v_a = a[0];
    size_t i_a = 0;
    for(size_t i=0; i < b_size; i++) {
      while ( a_max_order < b_max_order ) {
	i_a++;
	if( i_a < a_size ) {
	  v_a += (a[i_a] << (a_max_order - min_order));
	}
	a_max_order += bit_length_a;
      }
      b[i] = v_a & maxbit_b;
      v_a = v_a >> bit_length_b;
      min_order += bit_length_b;
      b_max_order += bit_length_b;
    }
  }

  /**
     Function same with previous change_bitlengh but for set of bitstrings
     @param[in] bit_length_a: the input length of bit string managed by each `size_t`
     @param[in/out] b: target set of bit strings
     @param[in] bit_length_b: target length of bit string managed by each `size_t`
   */
  void change_bitlength(size_t bit_length_a, std::vector<std::vector<size_t>> & b, size_t bit_length_b) {
    std::vector<size_t> a = b[0];
    size_t a_size = a.size();
    size_t total_bit_length_a = a_size * bit_length_a;
    size_t b_size = total_bit_length_a / bit_length_b;
    if( total_bit_length_a % bit_length_b != 0 ) {
      b_size++;
    }
    size_t maxbit_b;
    if (bit_length_b == 64)
      maxbit_b = 0xffffffffffffffff;
    else
      maxbit_b = (((size_t) 1) << bit_length_b) - 1;

    for(size_t k=0; k < b.size(); k++) {
      a = b[k];
      b[k].resize(b_size);

      size_t a_max_order = bit_length_a;
      size_t min_order = 0;
      size_t b_max_order = bit_length_b;
      size_t v_a = a[0];
      size_t i_a = 0;
      for(size_t i=0; i < b_size; i++) {
	while ( a_max_order < b_max_order ) {
	  i_a++;
	  if( i_a < a_size ) {
	    v_a += (a[i_a] << (a_max_order - min_order));
	  }
	  a_max_order += bit_length_a;
	}
	b[k][i] = (v_a & maxbit_b);
	v_a = (v_a >> bit_length_b);
	min_order += bit_length_b;
	b_max_order += bit_length_b;
      }
    }
  }

  /**
     Function to count the sign bit string
     @param[in] w: input bitstring
     @param[in] bit_length: the length of bit string managed by each `size_t` for `w`.
     @param[in] x: position of bit. Actual position corresponds to x + bit_length * r
     @param[in] r: position of bit.
   */
  inline int bit_string_sign_factor(const std::vector<size_t> & w,
				    int bit_length,
				    size_t x,
				    size_t r) {
    int sign = 1;
    size_t size_t_one = 1;
    for(size_t k=0; k < r; k++) {
      for(size_t l=0; l < bit_length; l++) {
	if( (w[k] & (size_t_one << l)) != 0 ) {
	  sign *= -1;
	}
      }
    }
    for(size_t l=0; l < x; l++) {
      if( ( w[r] & (size_t_one << l) ) != 0 ) {
	sign *= -1;
      }
    }
    return sign;
  }


  /**
     Function to change the integer to a string filled by zero from left
     @param[in] i: input integer
     @param[out] s: output string
   */
  void convert_int_to_string(int i, std::string & s) {
    std::string snum = std::to_string(i);
    if( i < 10 )
      {
	std::string zeros("0000000");
	s = zeros + snum;
      }
    else if( i < 100 )
      {
	std::string zeros("000000");
	s = zeros + snum;
      }
    else if( i < 1000 )
      {
	std::string zeros("00000");
	s = zeros + snum;
      }
    else if( i < 10000 )
      {
	std::string zeros("0000");
	s = zeros + snum;
      }
    else if( i < 100000 )
      {
	std::string zeros("000");
	s = zeros + snum;
      }
    else if( i < 1000000 )
      {
	std::string zeros("00");
	s = zeros + snum;
      }
    else if( i < 10000000 )
      {
	std::string zeros("0");
	s = zeros + snum;
      }
    else
      {
	s = snum;
      }
  }

  /**
     Function to get extension of the filename.
     @param[in] path: filename
   */
  inline std::string get_extension(const std::string &path) {
    std::string ext;
    size_t pos1 = path.rfind('.');
    if(pos1 != std::string::npos){
      ext = path.substr(pos1+1, path.size()-pos1);
      std::string::iterator itr = ext.begin();
      while(itr != ext.end()){
	*itr = tolower(*itr);
	itr++;
      }
      itr = ext.end()-1;
      while(itr != ext.begin()){
	if(*itr == 0 || *itr == 32){
	  ext.erase(itr--);
	}
	else{
	  itr--;
	}
      }
    }
    return ext;
  }

  /**
     Function to remove the extension of the filename.
     @param[in] filename: filename
   */
  std::string remove_extension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
  }

  /**
     Function to define the name of binary file with zero-filled integer
     @param[in] i: integer
     @param[in] filename: filename
   */
  std::string get_binary_file_name(int i, const std::string & filename) {
    std::string name = remove_extension(filename);
    std::string label;
    convert_int_to_string(i,label);
    std::string res = name + label + ".dat";
    return res;
  }

  /**
     Function to save the set of bitstring
     @param[in] os: output stream
     @param[in] config: the set of bitstrings
   */
  void SaveConfig(std::ostream & os,
		  std::vector<std::vector<size_t>> & config) {
    size_t size_v = config.size();
    size_t size_b = 0;
    if( config.size() != 0 ) {
      size_b = config[0].size();
    }
    os.write(reinterpret_cast<char *>(&size_v),sizeof(size_t));
    os.write(reinterpret_cast<char *>(&size_b),sizeof(size_t));
    for(size_t n=0; n < size_v; n++) {
      os.write(reinterpret_cast<char *>(config[n].data()),
	       sizeof(size_t)*size_b);
    }
  }

  /**
     Function to load the set of bitstring
     @param[in] is: input stream
     @param[in] config: the set of bitstrings
   */
  void LoadConfig(std::istream & is,
		  std::vector<std::vector<size_t>> & config) {
    size_t size_v;
    size_t size_b;
    is.read(reinterpret_cast<char *>(&size_v),sizeof(size_t));
    is.read(reinterpret_cast<char *>(&size_b),sizeof(size_t));
    config = std::vector<std::vector<size_t>>(size_v,std::vector<size_t>(size_b));
    for(size_t n=0; n < size_v; n++) {
      is.read(reinterpret_cast<char *>(config[n].data()),
	      sizeof(size_t)*size_b);
    }
  }

  /**
     structure for Hash function for sort: XOR base (while it has intersection risk, it is covered by equal)
   */
  struct BitVecHash {
    size_t operator()(const std::vector<size_t> & v) const {
      size_t hash = v.size();
      for (size_t x : v) {
	hash ^= x + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  /**
     Equal function: whether bit is equal or not
   */
  struct BitVecEqual {
    bool operator()(const std::vector<size_t> & a,
		    const std::vector<size_t>& b) const {
      return a == b; // std::vector<size_t>
    }
  };

  /**
     merge sequences D and W to Dn and Wn if D has same entries
     @param[in] D: input set of bit strings
     @param[in] W: weight for each bit string
     @param[out] Dn: unique set of bit strings
     @param[out] Wn: corresponding weights for each bit string
   */
  template<typename T>
  void merge_bit_sequences(
	 const std::vector<std::vector<size_t>> & D,
	 const std::vector<T> & W,
	 std::vector<std::vector<size_t>>& Dn,
	 std::vector<T>& Wn) {

    Dn = D;
    sort_bitarray(Dn);
    Wn.resize(Dn.size(),T(0.0));
    for(size_t i=0; i < D.size(); i++) {
      auto itn = std::lower_bound(Dn.begin(),Dn.end(),
				  D[i],
				  [](const std::vector<size_t> & x,
				     const std::vector<size_t> & y)
				  { return x < y; });
      size_t n = std::distance(Dn.begin(),itn);
      Wn[n] += W[i];
    }


  }


}

#endif
