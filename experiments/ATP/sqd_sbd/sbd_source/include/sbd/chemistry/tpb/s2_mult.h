/**
@file sbd/chemistry/tpb/s2_mult.h
@brief S^2 spin-squared penalty for Davidson solver.

Implements the full S^2 operator (diagonal + off-diagonal) for
use as a spin penalty: H_eff = H + shift * S^2.

For S_z = 0: S^2 = S_-S_+ = sum_{p,q} a+_{p,beta} a_{p,alpha} a+_{q,alpha} a_{q,beta}

Diagonal (p=q): S^2_diag(ia,ib) = N_beta - popcount(alpha[ia] AND beta[ib])
Off-diagonal (p!=q): connects determinants where alpha does p->q and beta does q->p
  Matrix element = (-1)^(alpha_between + beta_between) where
  alpha_between = # occupied alpha orbitals strictly between p and q
  beta_between  = # occupied beta orbitals strictly between p and q
*/
#ifndef SBD_CHEMISTRY_TPB_S2_MULT_H
#define SBD_CHEMISTRY_TPB_S2_MULT_H

namespace sbd {

/**
 * Precomputed sparse S^2 off-diagonal matrix in COO format.
 * The diagonal part is handled separately via hii modification.
 */
template <typename ElemT>
struct SparseS2 {
    std::vector<size_t> bra_idx;
    std::vector<size_t> ket_idx;
    std::vector<ElemT> values;   // shift * sign for each element
    ElemT shift = ElemT(0);

    /**
     * Apply off-diagonal S^2*C contribution: HC[bra] += val * C[ket]
     */
    void apply(const std::vector<ElemT>& C, std::vector<ElemT>& HC) const {
        if (std::abs(shift) < 1.0e-15 || bra_idx.empty()) return;
        for (size_t i = 0; i < bra_idx.size(); i++) {
            HC[bra_idx[i]] += values[i] * C[ket_idx[i]];
        }
    }

    size_t nnz() const { return bra_idx.size(); }
};


/**
 * Count occupied orbitals strictly between min(p,q)+1 and max(p,q)-1
 * in a half-determinant (alpha or beta) string.
 */
inline int count_occ_between(const std::vector<size_t>& det,
                              size_t bit_length, int p, int q) {
    int lo = std::min(p, q);
    int hi = std::max(p, q);
    int count = 0;
    for (int bit = lo + 1; bit < hi; bit++) {
        size_t word = bit / bit_length;
        size_t pos = bit % bit_length;
        if (det[word] & (size_t(1) << pos)) count++;
    }
    return count;
}


/**
 * Build the sparse S^2 off-diagonal matrix from helper structures.
 *
 * S^2 off-diagonal connects determinants where:
 *   alpha excitation: annihilate orbital p, create orbital q  (spin-orbs: an=2p, cr=2q)
 *   beta  excitation: annihilate orbital q, create orbital p  (spin-orbs: an=2q+1, cr=2p+1)
 *
 * In SBD helper notation, the S^2 filter condition is:
 *   cr_a + 1 == an_b  AND  an_a + 1 == cr_b
 *
 * @param shift S^2 penalty coefficient (e.g. 0.2)
 * @param adets Alpha half-determinant strings
 * @param bdets Beta half-determinant strings
 * @param bit_length Bits per word in determinant storage
 * @param norbs Number of spatial orbitals (L)
 * @param helper TaskHelpers from MakeHelpers
 * @param h_comm MPI communicator for Hamiltonian distribution
 * @return Precomputed sparse S^2 matrix
 */
template <typename ElemT>
SparseS2<ElemT> build_s2_offdiag(
        ElemT shift,
        const std::vector<std::vector<size_t>>& adets,
        const std::vector<std::vector<size_t>>& bdets,
        size_t bit_length,
        size_t norbs,
        const std::vector<TaskHelpers>& helper,
        MPI_Comm h_comm) {

    SparseS2<ElemT> s2;
    s2.shift = shift;
    if (std::abs(shift) < 1.0e-15 || helper.size() == 0) return s2;

    int mpi_rank_h = 0, mpi_size_h = 1;
    MPI_Comm_rank(h_comm, &mpi_rank_h);
    MPI_Comm_size(h_comm, &mpi_size_h);

    size_t braBetaSize = helper[0].braBetaEnd - helper[0].braBetaStart;

    for (size_t task = 0; task < helper.size(); task++) {
        if (helper[task].taskType != 0) continue;

        size_t ketBetaSize = helper[task].ketBetaEnd - helper[task].ketBetaStart;

        for (size_t ia = helper[task].braAlphaStart; ia < helper[task].braAlphaEnd; ia++) {
            size_t ia_local = ia - helper[task].braAlphaStart;

            for (size_t ib = helper[task].braBetaStart; ib < helper[task].braBetaEnd; ib++) {
                size_t braIdx = (ia - helper[task].braAlphaStart) * braBetaSize
                              + ib - helper[task].braBetaStart;
                if ((braIdx % mpi_size_h) != mpi_rank_h) continue;

                size_t ib_local = ib - helper[task].braBetaStart;

                for (size_t j = 0; j < helper[task].SinglesFromAlphaLen[ia_local]; j++) {
                    int cr_a = helper[task].SinglesAlphaCrAnSM[ia_local][2*j + 0];
                    int an_a = helper[task].SinglesAlphaCrAnSM[ia_local][2*j + 1];
                    size_t ja = helper[task].SinglesFromAlphaSM[ia_local][j];

                    for (size_t ki = 0; ki < helper[task].SinglesFromBetaLen[ib_local]; ki++) {
                        int cr_b = helper[task].SinglesBetaCrAnSM[ib_local][2*ki + 0];
                        int an_b = helper[task].SinglesBetaCrAnSM[ib_local][2*ki + 1];

                        // S^2 filter: complementary spin flips
                        // alpha p->q (an=2p, cr=2q) + beta q->p (an=2q+1, cr=2p+1)
                        if (cr_a + 1 != an_b || an_a + 1 != cr_b) continue;

                        size_t jb = helper[task].SinglesFromBetaSM[ib_local][ki];
                        size_t ketIdx = (ja - helper[task].ketAlphaStart) * ketBetaSize
                                      + jb - helper[task].ketBetaStart;

                        // Sign from half-det parities
                        // Alpha: annihilate orbital an_a/2, create orbital cr_a/2
                        int p_alpha = an_a / 2;
                        int q_alpha = cr_a / 2;
                        int alpha_between = count_occ_between(adets[ia], bit_length,
                                                              p_alpha, q_alpha);

                        // Beta: annihilate orbital an_b/2, create orbital cr_b/2
                        // an_b = 2q+1 -> orbital q, cr_b = 2p+1 -> orbital p
                        int p_beta = an_b / 2;
                        int q_beta = cr_b / 2;
                        int beta_between = count_occ_between(bdets[ib], bit_length,
                                                             p_beta, q_beta);

                        ElemT sign = ((alpha_between + beta_between) % 2 == 0)
                                   ? ElemT(1.0) : ElemT(-1.0);

                        s2.bra_idx.push_back(braIdx);
                        s2.ket_idx.push_back(ketIdx);
                        s2.values.push_back(shift * sign);
                    }
                }
            }
        }
    }

    return s2;
}

} // end namespace sbd

#endif
