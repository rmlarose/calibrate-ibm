//// This file is a part of sbd
/**
@file type_def.h
@brief Definitions of common types and related helper functions
*/
#ifndef SBD_FRAMEWORK_TYPE_DEF_H
#define SBD_FRAMEWORK_TYPE_DEF_H

#include <complex>
#include <limits.h>
#include <cassert>

#include "mpi.h"

namespace sbd {
  template <typename T> struct GetRealType;
  template <> struct GetRealType<float> { using RealT = float; };
  template <> struct GetRealType<double> { using RealT = double; };
  template <> struct GetRealType<std::complex<float>> { using RealT = float; };
  template <> struct GetRealType<std::complex<double>> { using RealT = double; };

  inline float GetReal(const float a) { return a; }
  inline double GetReal(const double a) { return a; }
  inline float GetReal(const std::complex<float> a) { return a.real(); }
  inline double GetReal(const std::complex<double> a) { return a.real(); }

#ifdef SBD_THRUST
  template <typename T> inline __host__ __device__ T Conjugate(T a) { return a; }
  template<> inline __host__ __device__ std::complex<float> Conjugate(std::complex<float> a) { return std::conj(a); }
  template<> inline __host__ __device__ std::complex<double> Conjugate(std::complex<double> a) { return std::conj(a); }
#else
  template <typename T> inline T Conjugate(T a) { return a; }
  template<> inline std::complex<float> Conjugate(std::complex<float> a) { return std::conj(a); }
  template<> inline std::complex<double> Conjugate(std::complex<double> a) { return std::conj(a); }
#endif

  template <typename T> struct GetMpiType { static MPI_Datatype MpiT; };
  template<> inline MPI_Datatype GetMpiType<float>::MpiT = MPI_FLOAT;
  template<> inline MPI_Datatype GetMpiType<double>::MpiT = MPI_DOUBLE;
  template<> inline MPI_Datatype GetMpiType<std::complex<float>>::MpiT = MPI_CXX_FLOAT_COMPLEX;
  template<> inline MPI_Datatype GetMpiType<std::complex<double>>::MpiT = MPI_CXX_DOUBLE_COMPLEX;

#if SIZE_MAX == UCHAR_MAX
  #define SBD_MPI_SIZE_T MPI_UNSIGNED_CHAR
  #define SBD_BIT_SIZE_T UCHAR_WIDTH
#elif SIZE_MAX == USHRT_MAX
  #define SBD_MPI_SIZE_T MPI_UNSIGNED_SHORT
  #define SBD_BIT_SIZE_T USHRT_WIDTH
#elif SIZE_MAX == UINT_MAX
  #define SBD_MPI_SIZE_T MPI_UNSIGNED
  #define SBD_BIT_SIZE_T UINT_WIDTH
#elif SIZE_MAX == ULONG_MAX
  #define SBD_MPI_SIZE_T MPI_UNSIGNED_LONG
  #define SBD_BIT_SIZE_T ULONG_WIDTH
#elif SIZE_MAX == ULLONG_MAX
  #define SBD_MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
  #define SBD_BIT_SIZE_T ULLONG_WIDTH
#else
   #error SIZE_MAX
#endif

} // end namespace sbd
#endif // end SBD_FRAMEWORK_TYPE_DEF_H
