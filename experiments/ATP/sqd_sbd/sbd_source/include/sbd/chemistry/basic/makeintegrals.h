/**
@file sbd/chemistry/basic/makeintegrals.h
@brief Setup oneInt and twoInt from fcidump format
*/
#ifndef SBD_CHEMISTRY_BASIC_MAKEINTEGRALS_H
#define SBD_CHEMISTRY_BASIC_MAKEINTEGRALS_H

namespace sbd {

  template <typename ElemT>
  void SetupIntegrals(const FCIDump & fcidump,
		      int & L,
		      int & N,
		      ElemT & I0,
		      oneInt<ElemT> & I1,
		      twoInt<ElemT> & I2) {

    for(const auto & [key, value] : fcidump.header) {
      if( key == std::string("NORB") ) {
	L = std::atoi(value.c_str());
      }
      if( key == std::string("NELEC") ) {
	N = std::atoi(value.c_str());
      }
    }

    I1.norbs = 2 * L;
    I1.store.resize(4*L*L,ElemT(0.0));

    int B = L*(L+1)/2;
    I2.norbs = L;
    I2.store.resize(B*(B+1)/2,ElemT(0.0));

    for(const auto & [value, i, j, k, l] : fcidump.integrals) {
      if( (i==0) && (k==0) && (j==0) && (l==0) ) {
	I0 = ElemT(value);
      } else if( (k==l) && (k==0) ) {
	I1(2*(i-1),2*(j-1)) = ElemT(value);
	I1(2*(i-1)+1,2*(j-1)+1) = ElemT(value);
	I1(2*(j-1),2*(i-1)) = ElemT(value);
	I1(2*(j-1)+1,2*(i-1)+1) = ElemT(value);
      } else {
	I2(2*(i-1),2*(j-1),2*(k-1),2*(l-1)) = ElemT(value);
      }
    }

    I2.DirectMat.resize(L*L);
    I2.ExchangeMat.resize(L*L);
    for(int i=0; i < L; i++) {
      for(int j=0; j < L; j++) {
	I2.Direct(i,j) = I2(2*i,2*i,2*j,2*j);
	I2.Exchange(i,j) = I2(2*i,2*j,2*j,2*i);
      }
    }
  }
  
}

#endif
