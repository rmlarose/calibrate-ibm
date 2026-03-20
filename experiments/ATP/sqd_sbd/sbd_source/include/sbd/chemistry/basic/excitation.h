/**
@file sbd/chemistry/basic/excitation.h
@brief function to find excitation from a determinant
 */
#ifndef SBD_CHEMISTRY_BASIC_EXCITATION_H
#define SBD_CHEMISTRY_BASIC_EXCITATION_H
namespace sbd {

  void single_from_hdet(const std::vector<size_t> & hdet_base,
			size_t bit_length,
			size_t norb,
			size_t num_closed,
			const std::vector<int> & open_base,
			const std::vector<int> & closed_base,
			std::vector<std::vector<size_t>> & hdet_ex) {
    // supporsed that open and closed are obtained priory by using getOpenClosed function for hdet_base.
    size_t num_ex = num_closed * (norb - num_closed);
    hdet_ex.resize(num_ex);
    size_t ex_count = 0;
    for(size_t j=0; j < num_closed; j++) {
      for(size_t k=0; k < norb-num_closed; k++) {
	hdet_ex[ex_count] = hdet_base;
	setocc(hdet_ex[ex_count],bit_length,closed_base[j],false);
	setocc(hdet_ex[ex_count],bit_length,open_base[k],true);
	ex_count++;
      }
    }
  }
  
  void single_from_hdet(const std::vector<size_t> & hdet,
			size_t bit_length,
			size_t norb,
			std::vector<std::vector<size_t>> & edet) {
    std::vector<int> open_base(norb);
    std::vector<int> closed_base(norb);
    int nc = getOpenClosed(hdet,bit_length,norb,open_base,closed_base);
    size_t numc = static_cast<size_t>(nc);
    single_from_hdet(hdet,bit_length,norb,numc,open_base,closed_base,edet);
  }
  
}

#endif
