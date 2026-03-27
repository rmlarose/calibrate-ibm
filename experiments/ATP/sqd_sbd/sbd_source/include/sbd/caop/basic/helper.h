/**
@file sbd/caop/basic/helpers.h
@brief Helper array to handle the Hamiltonian operations for distributed basis
*/
#ifndef SBD_CAOP_BASIC_HELPER_H
#define SBD_CAOP_BASIC_HELPER_H

#include <numeric>

namespace sbd {
  
  void setup_communicator(MPI_Comm comm,
			  int h_comm_size,
			  int b_comm_size,
			  int t_comm_size,
			  MPI_Comm & h_comm,
			  MPI_Comm & b_comm,
			  MPI_Comm & t_comm) {
    
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int a_comm_size = b_comm_size * t_comm_size;
    
    MPI_Comm a_comm;
    int a_comm_color = mpi_rank / a_comm_size;
    int h_comm_color = mpi_rank % a_comm_size;
    MPI_Comm_split(comm,a_comm_color,mpi_rank,&a_comm);
    MPI_Comm_split(comm,h_comm_color,mpi_rank,&h_comm);
    
    int mpi_size_a; MPI_Comm_size(a_comm,&mpi_size_a);
    int mpi_rank_a; MPI_Comm_rank(a_comm,&mpi_rank_a);
    
    int t_comm_color = mpi_rank_a % b_comm_size;
    int b_comm_color = mpi_rank_a / b_comm_size;
    MPI_Comm_split(a_comm,t_comm_color,mpi_rank,&t_comm);
    MPI_Comm_split(a_comm,b_comm_color,mpi_rank,&b_comm);
    
  }
  
  void make_slide(std::vector<int> & slide,
		  MPI_Comm & b_comm,
		  MPI_Comm & t_comm) {
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
    int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
    
    size_t shift_start = static_cast<size_t>(0);
    size_t shift_end   = static_cast<size_t>(mpi_size_b);
    get_mpi_range(mpi_size_t,mpi_rank_t,shift_start,shift_end);
    slide.resize(shift_end-shift_start);
    std::iota(slide.begin(),slide.end(),static_cast<int>(shift_start));
  }
  
}

#endif

