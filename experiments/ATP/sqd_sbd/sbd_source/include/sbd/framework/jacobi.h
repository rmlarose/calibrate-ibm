#ifndef SBD_FRAMEWORK_JACOBI_H
#define SBD_FRAMEWORK_JACOBI_H

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <type_traits>

namespace sbd {

  namespace hp_numeric {
    
    const double EPS_JACOBI = 1e-10;   // 
    const int MAX_ITER_JACOBI = 1000;  // 
    
    template <typename ElemT, typename RealT>
    void JacobiHeev(int N,
		    ElemT* A,
		    int LDA,
		    RealT* W) {
      
      int n = N;
      int lda = LDA;
      
      std::vector<ElemT> V(n * n, ElemT(0));
      for (int i = 0; i < n; ++i) V[i + i * n] = ElemT(1);

      std::vector<ElemT> B(n * n);
      
      
      int iter = 0;
      while (iter < MAX_ITER_JACOBI) {
	iter++;
	
	int p = 0, q = 1;
	RealT max_off_diag = 0.0;
	for (int i = 0; i < n; ++i) {
	  for (int j = i + 1; j < n; ++j) {
	    RealT abs_val = std::abs(A[i + j * lda]);
	    if (abs_val > max_off_diag) {
	      max_off_diag = abs_val;
	      p = i;
	      q = j;
	    }
	  }
	}
	if (max_off_diag < EPS_JACOBI) {
	  break;
	}

	RealT theta = RealT(0.5) * std::atan2(RealT(2) * std::real(A[p + q * lda]), std::real(A[q + q * lda]) - std::real(A[p + p * lda]));
	RealT c = std::cos(theta), s = std::sin(theta);

	for(int i=0; i < n; ++i) {
	  for(int j=0; j < n; ++j) {
	    B[i + j * n] = A[i + j * lda];
	  }
	}
	
	for(int k=0; k < n; k++) {
	  A[k + p * lda] = c * B[k + p * n] - s * B[k + q * n];
	  A[k + q * lda] = s * B[k + p * n] + c * B[k + q * n];
	  A[p + k * lda] = c * B[p + k * n] - s * B[q + k * n];
	  A[q + k * lda] = s * B[p + k * n] + c * B[q + k * n];
	}
	A[p + p * lda] = c * c * B[p + p * n] + s * s * B[q + q * n] - ElemT(2.0) * s * c * B[p + q * n];
	A[q + q * lda] = s * s * B[p + p * n] + c * c * B[q + q * n] + ElemT(2.0) * s * c * B[p + q * n];
	A[q + p * lda] = ElemT(0.0);
	A[p + q * lda] = ElemT(0.0);
	
	for (int k = 0; k < n; ++k) {
	  ElemT v_p = V[k + p * n];
	  ElemT v_q = V[k + q * n];
	  V[k + p * n] = c * v_p - s * v_q;
	  V[k + q * n] = s * v_p + c * v_q;
	}
      }
      
      if (iter >= MAX_ITER_JACOBI) {
	// *INFO = -1; // 収束しなかった
	return;
      }
      
      for (int i = 0; i < n; ++i) {
	W[i] = std::real(A[i + i * lda]);
      }
      
      for (int j = 0; j < n; ++j) {
	for (int i = 0; i < n; ++i) {
	  A[i + j * lda] = V[i + j * n];
	}
      }
    }
    
  } // end namespace hp_numeric
  
} // end namespace sbd

#endif
