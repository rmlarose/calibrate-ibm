/**
@file sbd/framework/fcidump.h
@brief fcidump format and readin
*/
#ifndef SBD_FRAMEWORK_FCIDUMP_H
#define SBD_FRAMEWORK_FCIDUMP_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <string>
#include <stdexcept>
#include <limits>
#include <iomanip>

namespace sbd {
  
  struct FCIDump {
    
    // Stores header information as key-value pairs
    std::map<std::string, std::string> header;
    // Stores integral data
    std::vector<std::tuple<double, int, int, int, int>> integrals;
    
  };

  FCIDump LoadFCIDump(const std::string & filename) {

    FCIDump fciDump;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
      throw std::runtime_error("Failed to open FCIDUMP file.");
    }
    
    std::string line;
    bool inHeader = true;
    std::string headerContent;
    
    while (std::getline(infile, line)) {
      // Skip empty lines or comments
      if (line.empty() || line[0] == '!') {
	continue;
      }
      
      if (inHeader) {

	// Append header lines to a single string until "&END" is found
	if (line.find("&END") != std::string::npos) {
	  inHeader = false; // End of header
	  continue;
	}
	
	size_t pos = line.find("&FCI");
	if (pos != std::string::npos) {
	  line.erase(pos, 4); // Remove "&FCI"
	}
	
	headerContent += line;
	
      } else {
	
	// Parse integral data
	std::stringstream ss(line);
	double value;
	int i, j, k, l;
	ss >> value >> i >> j >> k >> l;
	fciDump.integrals.emplace_back(value, i, j, k, l);
      }
    }
    
    // Parse the header content into key-value pairs
    std::stringstream headerStream(headerContent);
    std::string token;
    while (std::getline(headerStream, token, ',')) {
      size_t equalPos = token.find('=');
      if (equalPos != std::string::npos) {
	std::string key = token.substr(0, equalPos);
	std::string value = token.substr(equalPos + 1);
	
	// Trim whitespace
	key.erase(0, key.find_first_not_of(" \t"));
	key.erase(key.find_last_not_of(" \t") + 1);
	value.erase(0, value.find_first_not_of(" \t"));
	value.erase(value.find_last_not_of(" \t") + 1);
	
	fciDump.header[key] = value;
      }
    }
    
    infile.close();
    return fciDump;
    
    /*
    
    FCIDump fciDump;
    std::ifstream infile(filename);
    
    if (!infile.is_open()) {
      throw std::runtime_error("Failed to open FCIDUMP file.");
    }
    
    std::string line;
    bool inHeader = true;
    
    while (std::getline(infile, line)) {
      // Skip empty lines or comments
      if (line.empty() || line[0] == '!') {
	continue;
      }
      
      if (inHeader) {
	// Parse header lines
	if (line.find("&END") != std::string::npos) {
	  inHeader = false; // End of header
	  continue;
	}
	std::stringstream ss(line);
	std::string key, value;
	while (ss >> key >> value) {
	  if (key.back() == '=') {
	    key.pop_back(); // Remove '=' from key
	  }
	  fciDump.header[key] = value;
	}
      } else {
	// Parse integral data
	std::stringstream ss(line);
	double value;
	int i, j, k, l;
	ss >> value >> i >> j >> k >> l;
	fciDump.integrals.emplace_back(value, i, j, k, l);
      }
    }
    
    infile.close();
    return fciDump;
    */
  }

  // ---- Binary FCIDUMP format ----
  // Layout: [int32 NORB][int32 NELEC][int64 N][N × {double val, int32 i,j,k,l}]
  // Single fread() instead of per-line stringstream parsing.

  struct FCIDumpRecord {
    double value;
    int i, j, k, l;
  };

  void SaveFCIDumpBinary(const std::string& filename, const FCIDump& dump) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
      throw std::runtime_error("Failed to open binary FCIDUMP file for writing: " + filename);
    }
    int norb = std::atoi(dump.header.at("NORB").c_str());
    int nelec = std::atoi(dump.header.at("NELEC").c_str());
    ofs.write(reinterpret_cast<const char*>(&norb), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&nelec), sizeof(int));
    int64_t n = static_cast<int64_t>(dump.integrals.size());
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(int64_t));
    std::vector<FCIDumpRecord> records(n);
    for (int64_t idx = 0; idx < n; idx++) {
      records[idx] = {std::get<0>(dump.integrals[idx]),
                      std::get<1>(dump.integrals[idx]),
                      std::get<2>(dump.integrals[idx]),
                      std::get<3>(dump.integrals[idx]),
                      std::get<4>(dump.integrals[idx])};
    }
    ofs.write(reinterpret_cast<const char*>(records.data()),
              n * static_cast<int64_t>(sizeof(FCIDumpRecord)));
  }

  FCIDump LoadFCIDumpBinary(const std::string& filename) {
    FCIDump dump;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
      throw std::runtime_error("Failed to open binary FCIDUMP file: " + filename);
    }
    int norb, nelec;
    ifs.read(reinterpret_cast<char*>(&norb), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&nelec), sizeof(int));
    dump.header["NORB"] = std::to_string(norb);
    dump.header["NELEC"] = std::to_string(nelec);
    int64_t n;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(int64_t));
    std::vector<FCIDumpRecord> records(n);
    ifs.read(reinterpret_cast<char*>(records.data()),
             n * static_cast<int64_t>(sizeof(FCIDumpRecord)));
    dump.integrals.resize(n);
    for (int64_t idx = 0; idx < n; idx++) {
      dump.integrals[idx] = std::make_tuple(
        records[idx].value, records[idx].i, records[idx].j, records[idx].k, records[idx].l);
    }
    std::cout << "Loaded binary FCIDUMP: NORB=" << norb << " NELEC=" << nelec
              << " integrals=" << n << std::endl;
    return dump;
  }

  void printFCIDump(const FCIDump& fciDump) {
    // Print header
    std::cout << "Header:" << std::endl;
    for (const auto& [key, value] : fciDump.header) {
      std::cout << key << " = " << value << std::endl;
    }
    
    // Print integrals
    std::cout << "\nIntegrals:" << std::endl;
    for (const auto& [value, i, j, k, l] : fciDump.integrals) {
      std::cout << value << " " << i << " " << j << " " << k << " " << l << std::endl;
    }
  }

  // Serialize the FCIDump structure into a string for broadcasting
  std::string serializeFCIDump(const FCIDump& fciDump) {
    std::ostringstream oss;
    oss << std::scientific
	<< std::setprecision(std::numeric_limits<double>::max_digits10);
    // Serialize header
    for (const auto & [key, value] : fciDump.header) {
      oss << key << "=" << value << ",\n";
    }
    oss << "&END\n";
    
    // Serialize integrals
    for (const auto& [value, i, j, k, l] : fciDump.integrals) {
      oss << value << " " << i << " " << j << " " << k << " " << l << "\n";
    }
    return oss.str();
  }
  
  // Deserialize the string into an FCIDump structure
  FCIDump deserializeFCIDump(const std::string& data) {

    FCIDump fciDump;
    std::istringstream iss(data);
    std::string line;
    bool inHeader = true;
    std::string headerContent;
    
    while (std::getline(iss, line)) {
      if (line.empty()) {
	continue;
      }
      
      if (inHeader) {
	// end header
	if (line.find("&END") != std::string::npos) {
	  inHeader = false;
	  continue;
	}
	headerContent += line + " "; // combine header
      } else {
	// parse integral part
	std::stringstream ss(line);
	double value;
	int i, j, k, l;
	ss >> value >> i >> j >> k >> l;
	fciDump.integrals.emplace_back(value, i, j, k, l);
      }
    }

    // Analyze header part
    std::stringstream headerStream(headerContent);
    std::string token;
    while (std::getline(headerStream, token, ',')) {
      size_t equalPos = token.find('=');
      if (equalPos != std::string::npos) {
	std::string key = token.substr(0, equalPos);
	std::string value = token.substr(equalPos + 1);
	
	// Trim whitespace
	key.erase(0, key.find_first_not_of(" \t"));
	key.erase(key.find_last_not_of(" \t") + 1);
	value.erase(0, value.find_first_not_of(" \t"));
	value.erase(value.find_last_not_of(" \t") + 1);
	
	fciDump.header[key] = value;
      }
    }
    
    return fciDump;
    
  }


  void MpiBcast(FCIDump & fcidump,
		int root,
		MPI_Comm comm) {

    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int mpi_size; MPI_Comm_size(comm,&mpi_size);

    std::string serializeData;
    if( mpi_rank == root ) {
      serializeData = serializeFCIDump(fcidump);
    }
    
    // Broadcast the serialized data size
    int dataSize = serializeData.size();
    MPI_Bcast(&dataSize, 1, MPI_INT, root, comm);
    
    // Resize the buffer and broadcast the data
    serializeData.resize(dataSize);
    MPI_Bcast(&serializeData[0], dataSize, MPI_CHAR, root, comm);

    /*
    if( mpi_rank != 0 ) {
      fcidump = deserializeFCIDump(serializeData);
    }
    */
    fcidump = deserializeFCIDump(serializeData);
    
  }
  
  
} // end namespace sbd
#endif

