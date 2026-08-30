//// This file is a part of SBD
/**
@file hp_numeric.h
@brief Wrappers for high performance numerical function
*/
#ifndef SBD_FRAMEWORK_HP_NUMERIC_H
#define SBD_FRAMEWORK_HP_NUMERIC_H

#include "sbd/framework/type_def.h"

#include <array>
#include <string>
#include <string.h>

namespace sbd {
  namespace hp_numeric {

    extern "C" void sgemm_(char *, char *, int *, int *, int *,
			   float *, float *, int *, float *, int *,
			   float *, float *, int *);

    inline void MatGemm(char trans_a, char trans_b,
			int m, int n, int k,
			float alpha, float * pa, float * pb,
			float beta, float *pc) {
      int lda;
      if ((trans_a == 'N') || (trans_a == 'n')) {
	lda = m;
      } else {
	lda = k;
      }
      int ldb;
      if ((trans_b == 'N') || (trans_b == 'n')) {
	ldb = k;
      } else {
	ldb = n;
      }
      int ldc = m;
      sgemm_(&trans_a,&trans_b,&m,&n,&k,&alpha,pa,&lda,pb,&ldb,&beta,pc,&ldc);
    }

    extern "C" void dgemm_(char *, char *, int *, int *, int *,
			   double *, double *, int *, double *, int *,
			   double *, double *, int *);

    inline void MatGemm(char trans_a, char trans_b,
			int m, int n, int k,
			double alpha, double * pa, double * pb,
			double beta, double *pc) {
      int lda;
      if ((trans_a == 'N') || (trans_a == 'n')) {
	lda = m;
      } else {
	lda = k;
      }
      int ldb;
      if ((trans_b == 'N') || (trans_b == 'n')) {
	ldb = k;
      } else {
	ldb = n;
      }
      int ldc = m;
      dgemm_(&trans_a,&trans_b,&m,&n,&k,&alpha,pa,&lda,pb,&ldb,&beta,pc,&ldc);
    }

    extern "C" void cgemm_(char *, char *, int *, int *, int *,
			   std::complex<float> *, std::complex<float> *, int *, std::complex<float> *, int *,
			   std::complex<float> *, std::complex<float> *, int *);
    
    inline void MatGemm(char trans_a, char trans_b,
			int m, int n, int k,
			std::complex<float> alpha, std::complex<float> *pa, std::complex<float> *pb,
			std::complex<float> beta, std::complex<float> *pc) {
      int lda;
      if ((trans_a == 'N') || (trans_a == 'n')) {
	lda = m;
      } else {
	lda = k;
      }
      int ldb;
      if ((trans_b == 'N') || (trans_b == 'n')) {
	ldb = k;
      } else {
	ldb = n;
      }
      int ldc = m;
      cgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, pa, &lda, pb, &ldb, &beta, pc, &ldc);
    }

    extern "C" void zgemm_(char *, char *, int *, int *, int *,
			   std::complex<double> *, std::complex<double> *, int *, std::complex<double> *, int *,
			   std::complex<double> *, std::complex<double> *, int *);
    
    inline void MatGemm(char trans_a, char trans_b,
			int m, int n, int k,
			std::complex<double> alpha, std::complex<double> *pa, std::complex<double> *pb,
			std::complex<double> beta, std::complex<double> *pc) {
      int lda;
      if ((trans_a == 'N') || (trans_a == 'n')) {
	lda = m;
      } else {
	lda = k;
      }
      int ldb;
      if ((trans_b == 'N') || (trans_b == 'n')) {
	ldb = k;
      } else {
	ldb = n;
      }
      int ldc = m;
      zgemm_(&trans_a, &trans_b, &m, &n, &k, &alpha, pa, &lda, pb, &ldb, &beta, pc, &ldc);
    }

    // Matrix SVD
extern "C" void sgesdd_(
    char *, int *, int *,
    float *, int *, float *, float *, int *, float *, int *,
    float *, int *, int *, int *);

inline void MatSvd(
    char jobz, int m, int n, int k,
    float *pa, float *ps, float *pu, float *pvt) {
  int info;
  // Set workspace
  int *iwork = nullptr;
  int lwork = -1;
  float *work = (float *) malloc(sizeof(float));
  sgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, iwork, &info);    // query workspace
  assert(info == 0);                  // TODO: handle error
  lwork = work[0];                    // TODO:  float to int, check it is safe
  free(work);
  work = (float *) malloc(lwork * sizeof(float));
  iwork = (int *) malloc(8 * k * sizeof(int));
  // Actual calculation
  sgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, iwork, &info);
  assert(info == 0);                  // TODO: handle error
  free(work);
  free(iwork);
}

extern "C" void dgesdd_(
    char *, int *, int *,
    double *, int *, double *, double *, int *, double *, int *,
    double *, int *, int *, int *);

inline void MatSvd(
    char jobz, int m, int n, int k,
    double *pa, double *ps, double *pu, double *pvt) {
  int info;
  // Set workspace
  int *iwork = nullptr;
  int lwork = -1;
  double *work = (double *) malloc(sizeof(double));
  dgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, iwork, &info);    // query workspace
  assert(info == 0);                  // TODO: handle error
  lwork = work[0];                    // TODO:  float to int, check it is safe
  free(work);
  work = (double *) malloc(lwork * sizeof(double));
  iwork = (int *) malloc(8 * k * sizeof(int));
  // Actual calculation
  dgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, iwork, &info);
  assert(info == 0);                  // TODO: handle error
  free(work);
  free(iwork);
}

extern "C" void cgesdd_(
    char *, int *, int *,
    std::complex<float> *, int *, float *, std::complex<float> *, int *, std::complex<float> *, int *,
    std::complex<float> *, int *, float *, int *, int *);

inline void MatSvd(
    char jobz, int m, int n, int k,
    std::complex<float> *pa, float *ps, std::complex<float> *pu, std::complex<float> *pvt) {
  int info;
  // Set workspace
  int *iwork = nullptr;
  int lrwork;
  int mx = ((m > n) ? m : n);
  int mn = ((m < n) ? m : n);
  if (jobz == 'N') {
    lrwork = 5 * mn;        // LAPACK version <= 3.6 needs 7*mn, see LAPACK cgesdd doc
  } else if ((2*mx) > (3*mn + 4)) {
    if (mx > (10 * mn)) {   // in original LAPACK doc, mx >> mn
      lrwork = 5 * mn * mn + 5 * mn;
    } else {
      lrwork = 2 * mx * mn + 2 * mn * mn + mn;
    }
  } else {
    lrwork = 5 * mn * mn + 5 * mn;
  }
  float *rwork = nullptr;
  int lwork = -1;
  std::complex<float> *work = (std::complex<float> *) malloc(sizeof(std::complex<float>));
  cgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, rwork, iwork, &info);   // query workspace
  assert(info == 0);                        // TODO: handle error
  lwork = work[0].real();                   // TODO:  float to int, check it is safe
  free(work);
  work = (std::complex<float> *) malloc(lwork * sizeof(std::complex<float>));
  iwork = (int *) malloc(8 * k * sizeof(int));
  rwork = (float *) malloc(lrwork * sizeof(float));
  // Actual calculation
  cgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, rwork, iwork, &info);
  assert(info == 0);                        // TODO: handle error
  free(work);
  free(iwork);
  free(rwork);
}

extern "C" void zgesdd_(
    char *, int *, int *,
    std::complex<double> *, int *, double *, std::complex<double> *, int *, std::complex<double> *, int *,
    std::complex<double> *, int *, double *, int *, int *);

inline void MatSvd(
    char jobz, int m, int n, int k,
    std::complex<double> *pa, double *ps, std::complex<double> *pu, std::complex<double> *pvt) {
  int info;
  // Set workspace
  int *iwork = nullptr;
  int lrwork;
  int mx = ((m > n) ? m : n);
  int mn = ((m < n) ? m : n);
  if (jobz == 'N') {
    lrwork = 5 * mn;        // LAPACK version <= 3.6 needs 7*mn, see LAPACK cgesdd doc
  } else if ((2*mx) > (3*mn + 4)) {
    if (mx > (10 * mn)) {   // in original LAPACK doc, mx >> mn
      lrwork = 5 * mn * mn + 5 * mn;
    } else {
      lrwork = 2 * mx * mn + 2 * mn * mn + mn;
    }
  } else {
    lrwork = 5 * mn * mn + 5 * mn;
  }
  double *rwork = nullptr;
  int lwork = -1;
  std::complex<double> *work = (std::complex<double> *) malloc(sizeof(std::complex<double>));
  zgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, rwork, iwork, &info);   // query workspace
  lwork = work[0].real();                   // TODO: float to int, check it is safe
  assert(info == 0);                        // TODO: handle error
  free(work);
  work = (std::complex<double> *) malloc(lwork * sizeof(std::complex<double>));
  iwork = (int *) malloc(8 * k * sizeof(int));
  rwork = (double *) malloc(lrwork * sizeof(double));
  // Actual calculation
  zgesdd_(
      &jobz, &m, &n,
      pa, &m, ps, pu, &m, pvt, &k,
      work, &lwork, rwork, iwork, &info);
  assert(info == 0);                        // TODO: handle error
  free(work);
  free(iwork);
  free(rwork);
}


// Matrix QR
extern "C" void sgeqrf_(
    int *, int *, float *, int *, float *, float *, int *, int *);
extern "C" void sorgqr_(
    int *, int *, int *, float *, int *, float *, float *, int *, int *);

inline void MatQr(int m, int n, int k, float *pa, float *pq, float *pr) {
  int info;
  size_t elem_t_size = sizeof(float);
  // Calculate R matrix
  float *ptau = nullptr;
  // Set workspace
  int lwork = -1;
  float *work = (float *) malloc(elem_t_size);
  sgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0];    // TODO: float to int, check it is safe
  free(work);
  work = (float *) malloc(lwork * elem_t_size);
  // Actual calculation
  ptau = (float *) malloc(k * elem_t_size);
  sgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(work);
  // Set R elements
  for (int col_idx = 0; col_idx < n; ++col_idx) {
    if (col_idx < (k-1)) {
      memset(pr + col_idx * k + (col_idx + 1), 0, (k - 1 - col_idx) * elem_t_size);
      memcpy(pr + col_idx * k, pa + col_idx * m, (col_idx + 1) * elem_t_size);
    } else {
      memcpy(pr + col_idx * k, pa + col_idx * m, k * elem_t_size);
    }
  }

  // Calculate Q matrix
  // Set workspace
  lwork = -1;
  work = (float *) malloc(elem_t_size);
  sorgqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0];    // TODO: float to int, check it is safe
  free(work);
  work = (float *) malloc(lwork * elem_t_size);
  // Actual calculation
  memcpy(pq, pa, (m * k) * elem_t_size);
  sorgqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(ptau);
  free(work);
}


extern "C" void dgeqrf_(
    int *, int *, double *, int *, double *, double *, int *, int *);
extern "C" void dorgqr_(
    int *, int *, int *, double *, int *, double *, double *, int *, int *);

inline void MatQr(int m, int n, int k, double *pa, double *pq, double *pr) {
  int info;
  size_t elem_t_size = sizeof(double);
  // Calculate R matrix
  double *ptau = nullptr;
  // Set workspace
  int lwork = -1;
  double *work = (double *) malloc(elem_t_size);
  dgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0];    // TODO: float to int, check it is safe
  free(work);
  work = (double *) malloc(lwork * elem_t_size);
  // Actual calculation
  ptau = (double *) malloc(k * elem_t_size);
  dgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(work);
  // Set R elements
  for (int col_idx = 0; col_idx < n; ++col_idx) {
    if (col_idx < (k-1)) {
      memset(pr + col_idx * k + (col_idx + 1), 0, (k - 1 - col_idx) * elem_t_size);
      memcpy(pr + col_idx * k, pa + col_idx * m, (col_idx + 1) * elem_t_size);
    } else {
      memcpy(pr + col_idx * k, pa + col_idx * m, k * elem_t_size);
    }
  }

  // Calculate Q matrix
  // Set workspace
  lwork = -1;
  work = (double *) malloc(elem_t_size);
  dorgqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0];    // TODO: float to int, check it is safe
  free(work);
  work = (double *) malloc(lwork * elem_t_size);
  // Actual calculation
  memcpy(pq, pa, (m * k) * elem_t_size);
  dorgqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(ptau);
  free(work);
}


extern "C" void cgeqrf_(
    int *, int *, std::complex<float> *, int *, std::complex<float> *, std::complex<float> *, int *, int *);
extern "C" void cungqr_(
    int *, int *, int *, std::complex<float> *, int *, std::complex<float> *, std::complex<float> *, int *, int *);

inline void MatQr(int m, int n, int k, std::complex<float> *pa, std::complex<float> *pq, std::complex<float> *pr) {
  int info;
  size_t elem_t_size = sizeof(std::complex<float>);
  // Calculate R matrix
  std::complex<float> *ptau = nullptr;
  // Set workspace
  int lwork = -1;
  std::complex<float> *work = (std::complex<float> *) malloc(elem_t_size);
  cgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0].real();    // TODO: float to int, check it is safe
  free(work);
  work = (std::complex<float> *) malloc(lwork * elem_t_size);
  // Actual calculation
  ptau = (std::complex<float> *) malloc(k * elem_t_size);
  cgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(work);
  // Set R elements
  for (int col_idx = 0; col_idx < n; ++col_idx) {
    if (col_idx < (k-1)) {
      memset(pr + col_idx * k + (col_idx + 1), 0, (k - 1 - col_idx) * elem_t_size);
      memcpy(pr + col_idx * k, pa + col_idx * m, (col_idx + 1) * elem_t_size);
    } else {
      memcpy(pr + col_idx * k, pa + col_idx * m, k * elem_t_size);
    }
  }

  // Calculate Q matrix
  // Set workspace
  lwork = -1;
  work = (std::complex<float> *) malloc(elem_t_size);
  cungqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0].real();    // TODO: float to int, check it is safe
  free(work);
  work = (std::complex<float> *) malloc(lwork * elem_t_size);
  // Actual calculation
  memcpy(pq, pa, (m * k) * elem_t_size);
  cungqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(ptau);
  free(work);
}


extern "C" void zgeqrf_(
    int *, int *, std::complex<double> *, int *, std::complex<double> *, std::complex<double> *, int *, int *);
extern "C" void zungqr_(
    int *, int *, int *, std::complex<double> *, int *, std::complex<double> *, std::complex<double> *, int *, int *);

inline void MatQr(int m, int n, int k, std::complex<double> *pa, std::complex<double> *pq, std::complex<double> *pr) {
  int info;
  size_t elem_t_size = sizeof(std::complex<double>);
  // Calculate R matrix
  std::complex<double> *ptau = nullptr;
  // Set workspace
  int lwork = -1;
  std::complex<double> *work = (std::complex<double> *) malloc(elem_t_size);
  zgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0].real();    // TODO: float to int, check it is safe
  free(work);
  work = (std::complex<double> *) malloc(lwork * elem_t_size);
  // Actual calculation
  ptau = (std::complex<double> *) malloc(k * elem_t_size);
  zgeqrf_(&m, &n, pa, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(work);
  // Set R elements
  for (int col_idx = 0; col_idx < n; ++col_idx) {
    if (col_idx < (k-1)) {
      memset(pr + col_idx * k + (col_idx + 1), 0, (k - 1 - col_idx) * elem_t_size);
      memcpy(pr + col_idx * k, pa + col_idx * m, (col_idx + 1) * elem_t_size);
    } else {
      memcpy(pr + col_idx * k, pa + col_idx * m, k * elem_t_size);
    }
  }

  // Calculate Q matrix
  // Set workspace
  lwork = -1;
  work = (std::complex<double> *) malloc(elem_t_size);
  zungqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);   // query workspace
  lwork = work[0].real();    // TODO: float to int, check it is safe
  free(work);
  work = (std::complex<double> *) malloc(lwork * elem_t_size);
  // Actual calculation
  memcpy(pq, pa, (m * k) * elem_t_size);
  zungqr_(&m, &k, &k, pq, &m, ptau, work, &lwork, &info);
  assert(info == 0);
  free(ptau);
  free(work);
}


// Matrix diagonalization
// For hermitian(symmetric) matrix
    extern "C" void ssyev_(char *, char *, int *,
			   float *, int *, float *, float *, int *, int *);

    inline void MatHeev(char jobz, char uplo, int n, float *pa, int lda, float *pw) {
      int info;
      // Set workspace
      int lwork = -1;
      float *work = (float *) malloc(sizeof(float));
      ssyev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, &info);    // query workspace
      assert(info == 0);
      lwork = work[0];
      free(work);
      work = (float *) malloc(lwork * sizeof(float));
      // Actual calculation
      ssyev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, &info);
      assert(info == 0);
      free(work);
    }
    
    
    extern "C" void dsyev_(
			   char *, char *, int *, double *, int *, double *, double *, int *, int *);
    
    inline void MatHeev(char jobz, char uplo,
			int n, double *pa, int lda,
			double *pw) {
      int info;
      // Set workspace
      int lwork = -1;
      double *work = (double *) malloc(sizeof(double));
      dsyev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, &info);    // query workspace
      assert(info == 0);
      lwork = work[0];
      free(work);
      work = (double *) malloc(lwork * sizeof(double));
      // Actual calculation
      dsyev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, &info);
      assert(info == 0);
      free(work);
    }
    
    
    extern "C" void cheev_(
			   char *, char *, int *, std::complex<float> *, int *, float *, std::complex<float> *, int *, float *, int *);
    
    inline void MatHeev(char jobz, char uplo, int n, std::complex<float> *pa, int lda, float *pw) {
      int info;
      // Set workspace
      int lwork = -1;
      std::complex<float> *work = (std::complex<float> *) malloc(sizeof(std::complex<float>));
      int lrwork = ((1 > (3 * n - 2)) ? 1 : (3 * n - 2));
      float *rwork = (float *) malloc(lrwork * sizeof(float));
      cheev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, rwork, &info);    // query workspace
      assert(info == 0);
      lwork = work[0].real();
      free(work);
      work = (std::complex<float> *) malloc(lwork * sizeof(std::complex<float>));
      // Actual calculation
      cheev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, rwork, &info);
      assert(info == 0);
      free(work);
      free(rwork);
    }
    
    
    extern "C" void zheev_(
			   char *, char *, int *, std::complex<double> *, int *, double *, std::complex<double> *, int *, double *, int *);
    
    inline void MatHeev(char jobz, char uplo, int n, std::complex<double> *pa, int lda, double *pw) {
      int info;
      // Set workspace
      int lwork = -1;
      std::complex<double> *work = (std::complex<double> *) malloc(sizeof(std::complex<double>));
      int lrwork = ((1 > (3 * n - 2)) ? 1 : (3 * n - 2));
      double *rwork = (double *) malloc(lrwork * sizeof(double));
      zheev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, rwork, &info);    // query workspace
      assert(info == 0);
      lwork = work[0].real();
      free(work);
      work = (std::complex<double> *) malloc(lwork * sizeof(std::complex<double>));
      // Actual calculation
      zheev_(&jobz, &uplo, &n, pa, &lda, pw, work, &lwork, rwork, &info);
      assert(info == 0);
      free(work);
      free(rwork);
    }



    
			
    
  } // end namespace hp_numeric
} // end namespace sbd

#endif
