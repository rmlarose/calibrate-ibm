/**
@file sbd/caop/basic/basis.h
@brief function to setup the basis
 */
#ifndef SBD_CAOP_BASIC_BASIS_H
#define SBD_CAOP_BASIC_BASIS_H

#include <sys/stat.h>
#include <iomanip>

#include "sbd/framework/type_def.h"
#include "sbd/framework/mpi_utility.h"
#include "sbd/framework/bit_manipulation.h"

namespace sbd {
  
  void redistribution(std::vector<std::vector<size_t>> & config,
		      size_t bit_length,
		      size_t total_bit_length,
		      MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    std::vector<std::vector<size_t>> config_begin(mpi_size);
    std::vector<std::vector<size_t>> config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_redistribution(config,config_begin,config_end,index_begin,index_end,
		       total_bit_length,bit_length,comm);
    
  }
  
  void reordering(std::vector<std::vector<size_t>> & config,
		  size_t bit_length,
		  size_t total_bit_length,
		  MPI_Comm comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    std::vector<std::vector<size_t>> config_begin(mpi_size);
    std::vector<std::vector<size_t>> config_end(mpi_size);
    std::vector<size_t> index_begin(mpi_size);
    std::vector<size_t> index_end(mpi_size);
    mpi_sort_bitarray(config,config_begin,config_end,index_begin,index_end,
		      total_bit_length,bit_length,comm);
  }
  
  // I/O for basis
  void load_basis_from_file(const std::string & filename,
			    std::vector<std::vector<size_t>> & config,
			    size_t bit_length,
			    size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      std::ifstream ifs(filename);
      if( !ifs.is_open() ) {
	throw std::runtime_error("Failed to open basis bit-string file.");
      }
      std::string line;
      std::vector<std::string> lines;
      while( std::getline(ifs,line) ) {
	lines.push_back(line);
      }
      config.resize(lines.size());
      for(size_t i=0; i < lines.size(); i++) {
	config[i] = from_string(lines[i],bit_length,total_bit_length);
      }
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ifstream ifs(filename, std::ios::binary);
      if( !ifs.is_open() ) {
	throw std::runtime_error("Failed to open basis bit-string binary file.");
      }

      size_t inner_size = (total_bit_length+bit_length-1)/bit_length;
      ifs.seekg(0, std::ios::end);
      std::streampos file_size = ifs.tellg();
      ifs.seekg(0, std::ios::beg);

      size_t bytes_per_line = inner_size * sizeof(size_t);

      if( file_size % bytes_per_line != 0 ) {
	throw std::runtime_error("Binary file size mismatch");
      }

      size_t num_lines = file_size / bytes_per_line;

      config.resize(num_lines);
      for(size_t i=0; i < num_lines; i++) {
	config[i].resize(inner_size);
	ifs.read(reinterpret_cast<char*>(config[i].data()),bytes_per_line);
	if (!ifs) {
	  throw std::runtime_error("Failed to read binary basis data.");
	}
      }
    }
  }
  
  void save_basis_to_file(const std::string & filename,
			  std::vector<std::vector<size_t>> & config,
			  size_t bit_length,
			  size_t total_bit_length) {
    if( get_extension(filename) == std::string("txt") ) {
      std::ofstream ofs(filename);
      for(size_t i=0; i < config.size(); i++) {
	ofs << makestring(config[i],bit_length,total_bit_length) << std::endl;
      }
    } else if ( get_extension(filename) == std::string("bin") ) {
      std::ofstream ofs(filename,std::ios::binary);
      for(auto & b : config) {
	ofs.write(reinterpret_cast<char*>(b.data()),sizeof(size_t)*b.size());
      }
    }
  }

  // basis file name for multiple nodes
  std::string basisfilename(const std::string & basisname, int index, int filetype) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << index;
    std::string tag = oss.str();
    std::string filename;
    if( filetype == 0 ) {
      filename = basisname + tag + ".txt";
    } else if ( filetype == 1 ) {
      filename = basisname + tag + ".bin";
    }
    return filename;
  }

  void load_basis_from_files(const std::vector<std::string> & all_filenames,
			     std::vector<std::vector<size_t>> & config,
			     size_t bit_length,
			     size_t total_bit_length,
			     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    
    const int num_files = static_cast<int>(all_filenames.size());
    config.clear();
    
    if (num_files == 0) return;
    
    const int base = num_files / mpi_size;
    const int rem  = num_files % mpi_size;

    int my_first = 0;
    int my_count = 0;
    if (mpi_rank < rem) {
      my_count = base + 1;
      my_first = mpi_rank * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1) + (mpi_rank - rem) * base;
    }
    const int my_last = my_first + my_count;
    
    for (int i = my_first; i < my_last; ++i) {
      const std::string & fname = all_filenames[i];
      
      std::vector<std::vector<size_t>> local;
      load_basis_from_file(fname, local, bit_length, total_bit_length);
      
      config.insert(config.end(),
		    std::make_move_iterator(local.begin()),
		    std::make_move_iterator(local.end()));
    }
    sort_bitarray(config);
  }
  
  // load single file
  void load_basis_from_single_binary(const std::string & filename,
				     std::vector<std::vector<size_t>> & config,
				     size_t bit_length,
				     size_t total_bit_length,
				     MPI_Comm comm) {
    int mpi_rank; MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size; MPI_Comm_size(comm, &mpi_size);
    
    const size_t inner_size    = (total_bit_length + bit_length - 1) / bit_length;
    const size_t bytes_per_line = inner_size * sizeof(size_t);
    
    std::uint64_t num_lines_u64 = 0;
    
    if (mpi_rank == 0) {
      std::ifstream ifs(filename, std::ios::binary);
      if (!ifs.is_open()) {
	throw std::runtime_error("Failed to open basis binary file: " + filename);
      }
      
      ifs.seekg(0, std::ios::end);
      std::streampos file_size_pos = ifs.tellg();
      ifs.seekg(0, std::ios::beg);
      
    if (file_size_pos < 0) {
      throw std::runtime_error("tellg() failed for file: " + filename);
    }
    
    const std::uint64_t file_size = static_cast<std::uint64_t>(file_size_pos);
    
    if (file_size % bytes_per_line != 0) {
      throw std::runtime_error("Binary file size mismatch in " + filename);
    }
    
    num_lines_u64 = file_size / bytes_per_line;
    }
    
    MPI_Bcast(&num_lines_u64, 1, MPI_UINT64_T, 0, comm);

    if (num_lines_u64 == 0) {
      config.clear();
      return;
    }
    
    const std::size_t num_lines = static_cast<std::size_t>(num_lines_u64);

    const std::size_t base = num_lines / mpi_size;
    const std::size_t rem  = num_lines % mpi_size;
    
    std::size_t my_first = 0;
    std::size_t my_count = 0;
    if (static_cast<std::size_t>(mpi_rank) < rem) {
      my_count = base + 1;
      my_first = static_cast<std::size_t>(mpi_rank) * my_count;
    } else {
      my_count = base;
      my_first = rem * (base + 1)
	+ (static_cast<std::size_t>(mpi_rank) - rem) * base;
    }
    const std::size_t my_last = my_first + my_count;
    
    config.clear();
    config.resize(my_count);
    for (auto & row : config) {
      row.resize(inner_size);
    }
    
    if (my_count == 0) {
      return;
    }
    
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open basis binary file (per rank): " + filename);
    }

    const std::uint64_t my_offset_bytes =
      static_cast<std::uint64_t>(my_first) * bytes_per_line;
    
    ifs.seekg(static_cast<std::streamoff>(my_offset_bytes), std::ios::beg);
    if (!ifs) {
      throw std::runtime_error("seekg failed for basis binary file: " + filename);
    }
    
    for (std::size_t i = 0; i < my_count; ++i) {
      ifs.read(reinterpret_cast<char*>(config[i].data()), bytes_per_line);
      if (!ifs) {
	throw std::runtime_error("Failed to read basis data from: " + filename);
      }
    }
    sort_bitarray(config);
  }

  inline void mpi_bcast_string_vector(std::vector<std::string> & vec,
				      int root,
				      MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    int count = static_cast<int>(vec.size());
    MPI_Bcast(&count, 1, MPI_INT, root, comm);
    
    if (rank != root) {
      vec.resize(count);
    }
    
    std::vector<int> lengths(count);
    if (rank == root) {
      for (int i = 0; i < count; i++) {
	lengths[i] = static_cast<int>(vec[i].size());
      }
    }
    MPI_Bcast(lengths.data(), count, MPI_INT, root, comm);
    
    for (int i = 0; i < count; i++) {
      if (rank != root) {
	vec[i].resize(lengths[i]);
      }
      if (lengths[i] > 0) {
#ifdef SBD_TRADMODE
	char * ptr = &vec[i][0];
	MPI_Bcast(ptr, lengths[i], MPI_CHAR, root, comm);
#else
	MPI_Bcast(vec[i].data(), lengths[i], MPI_CHAR, root, comm);
#endif
      }
    }
  }
  
  
}

#endif
