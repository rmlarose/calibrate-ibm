/**
@file sbd/chemistry/basic/helpers.h
@brief Helper arrays to construct the Hamiltonian
*/
#ifndef SBD_CHEMISTRY_BASIC_HELPERS_H
#define SBD_CHEMISTRY_BASIC_HELPERS_H

#include <cstring>
#include <numeric>

namespace sbd {

  struct Helpers {
    std::vector<std::vector<size_t>> AlphaMajorToBeta;
    std::vector<std::vector<size_t>> AlphaMajorToDet;
    std::vector<std::vector<size_t>> BetaMajorToAlpha;
    std::vector<std::vector<size_t>> BetaMajorToDet;
    std::vector<std::vector<size_t>> SinglesFromAlpha;
    std::vector<std::vector<size_t>> SinglesFromBeta;
    std::map<std::vector<size_t>, size_t> BetaN;
    std::map<std::vector<size_t>, size_t> AlphaN;

    size_t* AlphaMajorToBetaLen;
    size_t* SinglesFromAlphaLen;
    size_t* BetaMajorToAlphaLen;
    size_t* SinglesFromBetaLen;

    std::vector<size_t*> AlphaMajorToBetaSM;
    std::vector<size_t*> AlphaMajorToDetSM;
    std::vector<size_t*> SinglesFromAlphaSM;
    std::vector<size_t*> BetaMajorToAlphaSM;
    std::vector<size_t*> BetaMajorToDetSM;
    std::vector<size_t*> SinglesFromBetaSM;
  };

  void updateAlphaBeta(std::vector<size_t> & DetAlpha,
		       std::vector<size_t> & DetBeta,
		       size_t bit_length,
		       size_t norb,
		       Helpers & helper,
		       std::vector<std::vector<size_t>> & Det,
		       size_t startIdx,
		       size_t DetIdx) {
    auto itb = helper.BetaN.find(DetBeta);
    if( itb == helper.BetaN.end() ) {
      auto ret = helper.BetaN.insert(std::pair<std::vector<size_t>,size_t>(
			DetBeta,helper.BetaMajorToDet.size()));
      itb = ret.first;
      helper.BetaMajorToAlpha.resize(itb->second+1);
      helper.BetaMajorToDet.resize(itb->second+1);
      helper.SinglesFromBeta.resize(itb->second+1);
      std::vector<int> closed(norb);
      std::vector<int> open(norb);
      int nclosed = getOpenClosed(DetBeta,bit_length,norb,open,closed);
      for(int j=0; j < nclosed; j++) {
	for(int k=0; k < norb-nclosed; k++) {
	  auto DetBetaCopy = DetBeta;
	  setocc(DetBetaCopy,bit_length,closed[j],false);
	  setocc(DetBetaCopy,bit_length,open[k],true);
	  auto itbcopy = helper.BetaN.find(DetBetaCopy);
	  if( itbcopy != helper.BetaN.end() ) {
	    helper.SinglesFromBeta[itb->second].push_back(itbcopy->second);
	    helper.SinglesFromBeta[itbcopy->second].push_back(itb->second);
	  }
	}
      }
    }
    
    auto ita = helper.AlphaN.find(DetAlpha);
    if( ita == helper.AlphaN.end() ) {
      auto ret = helper.AlphaN.insert(std::pair<std::vector<size_t>,size_t>(
			DetAlpha,helper.AlphaMajorToDet.size()));
      ita = ret.first;
      helper.AlphaMajorToBeta.resize(ita->second+1);
      helper.AlphaMajorToDet.resize(ita->second+1);
      helper.SinglesFromAlpha.resize(ita->second+1);
      std::vector<int> closed(norb);
      std::vector<int> open(norb);
      int nclosed = getOpenClosed(DetAlpha,bit_length,norb,open,closed);
      for(int j=0; j < nclosed; j++) {
	for(int k=0; k < norb-nclosed; k++) {
	  auto DetAlphaCopy = DetAlpha;
	  setocc(DetAlphaCopy,bit_length,closed[j],false);
	  setocc(DetAlphaCopy,bit_length,open[k],true);
	  auto itacopy = helper.AlphaN.find(DetAlphaCopy);
	  if( itacopy != helper.AlphaN.end() ) {
	    helper.SinglesFromAlpha[ita->second].push_back(itacopy->second);
	    helper.SinglesFromAlpha[itacopy->second].push_back(ita->second);
	  }
	}
      }
    }

    helper.AlphaMajorToBeta[ita->second].push_back(itb->second);
    helper.AlphaMajorToDet[ita->second].push_back(DetIdx);
    helper.BetaMajorToAlpha[itb->second].push_back(ita->second);
    helper.BetaMajorToDet[itb->second].push_back(DetIdx);
    
  }

  void PopulateHelpers(std::vector<std::vector<size_t>> & Det,
		       size_t bit_length,
		       size_t norb,
		       size_t startIdx,
		       Helpers & helper) {

    for(size_t i=startIdx; i < Det.size(); i++) {
      auto DetAlpha = getAlpha(Det[i],bit_length,norb);
      auto DetBeta = getBeta(Det[i],bit_length,norb);
      updateAlphaBeta(DetAlpha,DetBeta,bit_length,norb,
		      helper,Det,startIdx,i+1);
    }
    for(size_t i=0; i < helper.AlphaMajorToBeta.size(); i++) {
      
      std::vector<size_t> & betaRef = helper.AlphaMajorToBeta[i];
      std::vector<size_t> & detRef = helper.AlphaMajorToDet[i];
      std::vector<size_t> detIndex(betaRef.size());
      std::iota(detIndex.begin(),detIndex.end(),0);
      std::sort(detIndex.begin(),detIndex.end(),
		[&](size_t lhs, size_t rhs) {
		    return betaRef[lhs] < betaRef[rhs]; } );
      std::vector<size_t> sortedBeta(betaRef.size());
      std::vector<size_t> sortedDet(detRef.size());
      for(size_t j=0; j < detIndex.size(); j++) {
	sortedBeta[j] = betaRef[detIndex[j]];
	sortedDet[j] = detRef[detIndex[j]];
      }
      betaRef = std::move(sortedBeta);
      detRef = std::move(sortedDet);
      
      std::sort(helper.SinglesFromAlpha[i].begin(),
		helper.SinglesFromAlpha[i].end());
    }
    
    for(size_t i=0; i < helper.BetaMajorToAlpha.size(); i++) {
      std::vector<size_t> & alphaRef = helper.BetaMajorToAlpha[i];
      std::vector<size_t> & detRef = helper.BetaMajorToDet[i];
      std::vector<size_t> detIndex(alphaRef.size());
      std::iota(detIndex.begin(),detIndex.end(),0);
      std::sort(detIndex.begin(),detIndex.end(),
		[&](size_t lhs, size_t rhs) {
		  return alphaRef[lhs] < alphaRef[rhs]; } );
      std::vector<size_t> sortedAlpha(alphaRef.size());
      std::vector<size_t> sortedDet(detRef.size());
      for(size_t j=0; j < detIndex.size(); j++) {
	sortedAlpha[j] = alphaRef[detIndex[j]];
	sortedDet[j] = detRef[detIndex[j]];
      }
      alphaRef = std::move(sortedAlpha);
      detRef = std::move(sortedDet);

      std::sort(helper.SinglesFromBeta[i].begin(),
		helper.SinglesFromBeta[i].end());
    }
  }
  
  void MakeHelper(Helpers & helper, std::vector<size_t> & sharedMemory) {
    size_t nAlpha = helper.AlphaMajorToBeta.size();
    size_t nBeta = helper.BetaMajorToAlpha.size();
    
    helper.AlphaMajorToBetaLen = (size_t*)malloc(nAlpha * sizeof(size_t));
    helper.SinglesFromAlphaLen = (size_t*)malloc(nAlpha * sizeof(size_t));
    helper.BetaMajorToAlphaLen = (size_t*)malloc(nBeta * sizeof(size_t));
    helper.SinglesFromBetaLen = (size_t*)malloc(nBeta * sizeof(size_t));
    
    for (size_t i = 0; i < nAlpha; ++i) {
      helper.AlphaMajorToBetaLen[i] = helper.AlphaMajorToBeta[i].size();
      helper.SinglesFromAlphaLen[i] = helper.SinglesFromAlpha[i].size();
    }
    for (size_t i = 0; i < nBeta; ++i) {
      helper.BetaMajorToAlphaLen[i] = helper.BetaMajorToAlpha[i].size();
      helper.SinglesFromBetaLen[i] = helper.SinglesFromBeta[i].size();
    }

    helper.AlphaMajorToBetaSM.resize(nAlpha);
    helper.AlphaMajorToDetSM.resize(nAlpha);
    helper.SinglesFromAlphaSM.resize(nAlpha);

    helper.BetaMajorToAlphaSM.resize(nBeta);
    helper.BetaMajorToDetSM.resize(nBeta);
    helper.SinglesFromBetaSM.resize(nBeta);

    size_t total_size = 0;
    for (size_t i = 0; i < nAlpha; i++) {
      total_size += 2 * helper.AlphaMajorToBetaLen[i]
	              + helper.SinglesFromAlphaLen[i];
    }
    for (size_t i = 0; i < nBeta; i++) {
      total_size += 2 * helper.BetaMajorToAlphaLen[i]
	              + helper.SinglesFromBetaLen[i];
    }

    sharedMemory.resize(total_size);

    // size_t * begin = (size_t *) malloc(total_size * sizeof(size_t));
    // size_t * begin = helper.SinglesFromBetaLen + nBeta;
    size_t * begin = sharedMemory.data();
    size_t counter = 0;

    for (size_t i = 0; i < nAlpha; i++) {
      helper.AlphaMajorToBetaSM[i] = begin + counter;
      counter += helper.AlphaMajorToBetaLen[i];
      helper.AlphaMajorToDetSM[i] = begin + counter;
      counter += helper.AlphaMajorToBetaLen[i];
      helper.SinglesFromAlphaSM[i] = begin + counter;
      counter += helper.SinglesFromAlphaLen[i];
    }
    
    for (size_t i = 0; i < nBeta; i++) {
      helper.BetaMajorToAlphaSM[i] = begin + counter;
      counter += helper.BetaMajorToAlphaLen[i];
      helper.BetaMajorToDetSM[i] = begin + counter;
      counter += helper.BetaMajorToAlphaLen[i];
      helper.SinglesFromBetaSM[i] = begin + counter;
      counter += helper.SinglesFromBetaLen[i];
    }
    
    for (size_t i = 0; i < nAlpha; i++) {
      std::memcpy(helper.AlphaMajorToBetaSM[i],
		  helper.AlphaMajorToBeta[i].data(),
		  helper.AlphaMajorToBetaLen[i] * sizeof(size_t));
      std::memcpy(helper.AlphaMajorToDetSM[i],
		  helper.AlphaMajorToDet[i].data(),
		  helper.AlphaMajorToBetaLen[i] * sizeof(size_t));
      std::memcpy(helper.SinglesFromAlphaSM[i],
		  helper.SinglesFromAlpha[i].data(),
		  helper.SinglesFromAlphaLen[i] * sizeof(size_t));
    }
    for (size_t i = 0; i < nBeta; i++) {
      std::memcpy(helper.BetaMajorToAlphaSM[i],
		  helper.BetaMajorToAlpha[i].data(),
		  helper.BetaMajorToAlphaLen[i] * sizeof(size_t));
      std::memcpy(helper.BetaMajorToDetSM[i],
		  helper.BetaMajorToDet[i].data(),
		  helper.BetaMajorToAlphaLen[i] * sizeof(size_t));
      std::memcpy(helper.SinglesFromBetaSM[i],
		  helper.SinglesFromBeta[i].data(),
		  helper.SinglesFromBetaLen[i] * sizeof(size_t));
    }
  }

}

#endif // end SBD_CHEMISTRY_HELPERS_H
