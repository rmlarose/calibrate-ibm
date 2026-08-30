/**
@file sbd/framework/ssutils.h
@brief Utility for sparse-solver
*/
#ifndef SBD_FRAMEWORK_SSUTILS_H
#define SBD_FRAMEWORK_SSUTILS_H

namespace sbd {

  template <typename ElemT>
  void GetTotalD(const std::vector<ElemT> & hii,
		 std::vector<ElemT> & dii,
		 MPI_Comm h_comm) {
    size_t size_d = hii.size();
    dii.resize(static_cast<size_t>(size_d),ElemT(0.0));
    MPI_Datatype DataT = GetMpiType<ElemT>::MpiT;
#if MPI_VERSION >= 4
    MPI_Allreduce_c(hii.data(),dii.data(),size_d,DataT,MPI_SUM,h_comm);
#else
    MPI_Allreduce(hii.data(),dii.data(),size_d,DataT,MPI_SUM,h_comm);
#endif
  }
  
} // end namespace sbd

#endif // endif for SBD_HCBOSON_OUT_OF_PLACE_FUNC_DAVIDSON_H

