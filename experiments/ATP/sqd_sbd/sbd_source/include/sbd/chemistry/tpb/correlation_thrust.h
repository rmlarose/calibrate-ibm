/**
@file sbd/chemistry/tpb/correlation_thrust.h
@brief function to evaluate correlation functions ( < cdag cdag c c > and < cdag c > ) in general
*/
#ifndef SBD_CHEMISTRY_TPB_CORRELATION_THRUST_H
#define SBD_CHEMISTRY_TPB_CORRELATION_THRUST_H

namespace sbd
{

template <typename ElemT>
class CorrelationKernelBase : public MultKernelBase<ElemT> {
protected:
    ElemT* onebody;
    ElemT* twobody;
    size_t onebody_size;
    size_t twobody_size;
public:
    CorrelationKernelBase() {}

    CorrelationKernelBase(const MultDataThrust<ElemT>& data,
                        const thrust::device_vector<ElemT>& v_wb,
                        const thrust::device_vector<ElemT>& v_t,
                        thrust::device_vector<ElemT>& b1,
                        thrust::device_vector<ElemT>& b2) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        onebody = (ElemT*)thrust::raw_pointer_cast(b1.data());
        twobody = (ElemT*)thrust::raw_pointer_cast(b2.data());
        onebody_size = this->norbs * this->norbs;
        twobody_size = this->norbs * this->norbs * this->norbs * this->norbs;
    }

    /**
         Function for adding diagonal contribution
    */
    __device__ __host__ void ZeroDiffCorrelation(const size_t* det, ElemT WeightI)
    {
        for (int i = 0; i < 2 * this->norbs; i++) {
            if (this->getocc(det, i)) {
                int oi = i / 2;
                int si = i % 2;
                atomicAdd(onebody + si * onebody_size + oi + this->norbs * oi, Conjugate(WeightI) * WeightI);
                for (int j = i + 1; j < 2 * this->norbs; j++) {
                    if (this->getocc(det, j)) {
                        int oj = j / 2;
                        int sj = j % 2;
                        atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oi + this->norbs * oj + this->norbs * this->norbs * oi + this->norbs * this->norbs * this->norbs * oj), Conjugate(WeightI) * WeightI);
                        atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + this->norbs * oi + this->norbs * this->norbs * oj + this->norbs * this->norbs * this->norbs * oi), Conjugate(WeightI) * WeightI);
                        if (si == sj) {
                            atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oi + this->norbs * oj + this->norbs * this->norbs * oj + this->norbs * this->norbs * this->norbs * oi), -Conjugate(WeightI) * WeightI);
                            atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + this->norbs * oi + this->norbs * this->norbs * oi + this->norbs * this->norbs * this->norbs * oj), -Conjugate(WeightI) * WeightI);
                        }
                    }
                }
            }
        }
    }

    /**
        Function for adding one-occupation different contribution
    */
    __device__ __host__ void OneDiffCorrelation(const size_t* det,
                            const ElemT WeightI,
                            const ElemT WeightJ,
                            int i,
                            int a)
    {
        double sgn = 1.0;
        this->parity(det, std::min(i, a), std::max(i, a), sgn);
        int oi = i / 2;
        int si = i % 2;
        int oa = a / 2;
        int sa = a % 2;
        atomicAdd(onebody + si * onebody_size + (oi + this->norbs * oa), Conjugate(WeightI) * WeightJ * ElemT(sgn));
        size_t one = 1;
        for (int x = 0; x < this->size_D; x++) {
            size_t bits = det[x];
            for (int pos = 0; pos < this->bit_length; pos++) {
                if ((bits & 1ULL) == 1ULL) {
                    int soj = x * this->bit_length + pos;
                    int oj = soj / 2;
                    int sj = soj % 2;

                    atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + oj * this->norbs + oi * this->norbs * this->norbs + oj * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(sgn));
                    atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + oa * this->norbs + oj * this->norbs * this->norbs + oi * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(sgn));

                    if (si == sj) {
                        atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + oj * this->norbs + oj * this->norbs * this->norbs + oi * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(-sgn));
                        atomicAdd(twobody + (sj + 2 * si) * twobody_size + (oj + oa * this->norbs + oi * this->norbs * this->norbs + oj * this->norbs * this->norbs * this->norbs), Conjugate(WeightI) * WeightJ * ElemT(-sgn));
                    }
                }
                bits >>= 1;
            }
        }
    }

    /**
        Function for adding two-occupation different contribution
    */
    __device__ __host__ void TwoDiffCorrelation(const size_t* det,
                            const ElemT WeightI,
                            const ElemT WeightJ,
                            int i,
                            int j,
                            int a,
                            int b)
    {
        double sgn = 1.0;
        int I = std::min(i, j);
        int J = std::max(i, j);
        int A = std::min(a, b);
        int B = std::max(a, b);
        this->parity(det, std::min(I, A), std::max(I, A), sgn);
        this->parity(det, std::min(J, B), std::max(J, B), sgn);
        if (A > J || B < I)
            sgn *= -1.0;
        int oi = I / 2;
        int si = I % 2;
        int oa = A / 2;
        int sa = A % 2;
        int oj = J / 2;
        int sj = J % 2;
        int ob = B / 2;
        int sb = B % 2;

        if (si == sa) {
            atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + this->norbs * ob + this->norbs * this->norbs * (oi + this->norbs * oj)), ElemT(sgn) * Conjugate(WeightI) * WeightJ);
            atomicAdd(twobody + (sj + 2 * si) * twobody_size + (ob + this->norbs * oa + this->norbs * this->norbs * (oj + this->norbs * oi)), ElemT(sgn) * Conjugate(WeightI) * WeightJ);
        }

        if (si == sb) {
            atomicAdd(twobody + (si + 2 * sj) * twobody_size + (oa + this->norbs * ob + this->norbs * this->norbs * (oj + this->norbs * oi)), ElemT(-sgn) * Conjugate(WeightI) * WeightJ);
            atomicAdd(twobody + (sj + 2 * si) * twobody_size + (ob + this->norbs * oa + this->norbs * this->norbs * (oi + this->norbs * oj)), ElemT(-sgn) * Conjugate(WeightI) * WeightJ);
        }
    }
};

template <typename ElemT>
class CorrelationInit : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationInit(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;

        if (i + offset < braAlphaSize * braBetaSize) {
            if( ((i + offset) % this->mpi_size_h) == this->mpi_rank_h ) {
                size_t* DetI = this->det_I + (i + offset) * this->size_D;
                this->ZeroDiffCorrelation(DetI, this->Wb[i + offset]);
            }
        }
    }
};

template <typename ElemT>
class CorrelationInitNoCache : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationInitNoCache(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI = this->det_I + i * this->size_D;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);
            this->ZeroDiffCorrelation(DetI, this->Wb[braIdx]);
        }
    }
};


template <typename ElemT>
class CorrelationAlphaBeta : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationAlphaBeta(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        if (i + offset < helper.size_single_alpha * helper.size_single_beta) {
            size_t j = (i + offset) / helper.size_single_beta;
            size_t k = (i + offset) - j * helper.size_single_beta;

            size_t ia = helper.SinglesFromAlphaBraIndex[j];
            size_t ja = helper.SinglesFromAlphaKetIndex[j];
            size_t ib = helper.SinglesFromBetaBraIndex[k];
            size_t jb = helper.SinglesFromBetaKetIndex[k];

            size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
            if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;

                size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->size_D;
                ElemT WeightI = this->Wb[braIdx];
                ElemT WeightJ = this->T[ketIdx];

                this->TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                    helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                                    helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
            }
        }
    }
};


template <typename ElemT>
class CorrelationSingleAlpha : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationSingleAlpha(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        if (i + offset < helper.size_single_alpha * (helper.braBetaEnd - helper.braBetaStart)) {
            size_t k = (i + offset) / helper.size_single_alpha;
            size_t j = (i + offset) - k * helper.size_single_alpha;

            size_t ia = helper.SinglesFromAlphaBraIndex[j];
            size_t ja = helper.SinglesFromAlphaKetIndex[j];
            size_t ib = k + helper.braBetaStart;
            size_t jb = ib;
            size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
            if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;

                size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->size_D;
                ElemT WeightI = this->Wb[braIdx];
                ElemT WeightJ = this->T[ketIdx];
                this->OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha]);
            }
        }
    }
};

template <typename ElemT>
class CorrelationDoubleAlpha : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationDoubleAlpha(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        if (i + offset < helper.size_double_alpha * (helper.braBetaEnd - helper.braBetaStart)) {
            size_t k = (i + offset) / helper.size_double_alpha;
            size_t j = (i + offset) - k * helper.size_double_alpha;

            size_t ia = helper.DoublesFromAlphaBraIndex[j];
            size_t ja = helper.DoublesFromAlphaKetIndex[j];
            size_t ib = k + helper.braBetaStart;
            size_t jb = ib;
            size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
            if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;

                size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->size_D;
                ElemT WeightI = this->Wb[braIdx];
                ElemT WeightJ = this->T[ketIdx];

                this->TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                    helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_alpha],
                                    helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_alpha], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_alpha]);
            }
        }
    }
};

template <typename ElemT>
class CorrelationSingleBeta : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationSingleBeta(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        if (i + offset < helper.size_single_beta * (helper.braAlphaEnd - helper.braAlphaStart)) {
            size_t j = (i + offset) / helper.size_single_beta;
            size_t k = (i + offset) - j * helper.size_single_beta;

            size_t ia = j + helper.braAlphaStart;
            size_t ja = ia;
            size_t ib = helper.SinglesFromBetaBraIndex[k];
            size_t jb = helper.SinglesFromBetaKetIndex[k];
            size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
            if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;

                size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->size_D;
                ElemT WeightI = this->Wb[braIdx];
                ElemT WeightJ = this->T[ketIdx];
                this->OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
            }
        }
    }
};

template <typename ElemT>
class CorrelationDoubleBeta : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationDoubleBeta(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        if (i + offset < helper.size_double_beta * (helper.braAlphaEnd - helper.braAlphaStart)) {
            size_t j = (i + offset) / helper.size_double_beta;
            size_t k = (i + offset) - j * helper.size_double_beta;

            size_t ia = j + helper.braAlphaStart;
            size_t ja = ia;
            size_t ib = helper.DoublesFromBetaBraIndex[k];
            size_t jb = helper.DoublesFromBetaKetIndex[k];
            size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
            if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;

                size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * this->bdets_size + ib - helper.braBetaStart) * this->size_D;
                ElemT WeightI = this->Wb[braIdx];
                ElemT WeightJ = this->T[ketIdx];

                this->TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                            helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_beta],
                                            helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_beta], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_beta]);
            }
        }
    }
};


template <typename ElemT>
class CorrelationTask0 : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationTask0(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI = this->det_I + i * this->size_D;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);
            ElemT WeightI = this->Wb[braIdx];

            for (size_t j = helper.SinglesFromAlphaOffset[a]; j < helper.SinglesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                for (size_t k = helper.SinglesFromBetaOffset[b]; k < helper.SinglesFromBetaOffset[b + 1]; k++) {
                    size_t jb = helper.SinglesFromBetaKetIndex[k];
                    size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                    + jb - helper.ketBetaStart;
                    ElemT WeightJ = this->T[ketIdx];
                    this->TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                        helper.SinglesAlphaCrAnSM[j], helper.SinglesBetaCrAnSM[k],
                                        helper.SinglesAlphaCrAnSM[j + helper.size_single_alpha], helper.SinglesBetaCrAnSM[k + helper.size_single_beta]);
                }
            }
        }
    }
};

template <typename ElemT>
class CorrelationTask1 : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationTask1(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI = this->det_I + i * this->size_D;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);
            ElemT WeightI = this->Wb[braIdx];

            for (size_t k = helper.SinglesFromBetaOffset[b]; k < helper.SinglesFromBetaOffset[b + 1]; k++) {
                size_t ja = ia;
                size_t jb = helper.SinglesFromBetaKetIndex[k];
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesBetaCrAnSM[k], helper.SinglesBetaCrAnSM[k + helper.size_single_alpha]);
            }
            for (size_t k = helper.DoublesFromBetaOffset[b]; k < helper.DoublesFromBetaOffset[b + 1]; k++) {
                size_t ja = ia;
                size_t jb = helper.DoublesFromBetaKetIndex[k];
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                        helper.DoublesBetaCrAnSM[k], helper.DoublesBetaCrAnSM[k + helper.size_double_alpha],
                                        helper.DoublesBetaCrAnSM[k + 2 * helper.size_double_alpha], helper.DoublesBetaCrAnSM[k + 3 * helper.size_double_alpha]);
           }
        }
    }
};

template <typename ElemT>
class CorrelationTask2 : public CorrelationKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    CorrelationTask2(const TaskHelpersThrust<ElemT>& h,
                const MultDataThrust<ElemT>& data,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                thrust::device_vector<ElemT>& b1,
                thrust::device_vector<ElemT>& b2,
                size_t o ) : CorrelationKernelBase<ElemT>(data, v_wb, v_t, b1, b2)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t braIdx = i + offset;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        size_t a = braIdx / braBetaSize;
        size_t b = braIdx - a * braBetaSize;
        size_t* DetI = this->det_I + i * this->size_D;
        size_t ia = a + helper.braAlphaStart;
        size_t ib = b + helper.braBetaStart;

        if ((braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);
            ElemT WeightI = this->Wb[braIdx];

            for (size_t j = helper.SinglesFromAlphaOffset[a]; j < helper.SinglesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.SinglesFromAlphaKetIndex[j];
                size_t jb = ib;
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->OneDiffCorrelation(DetI, WeightI, WeightJ, helper.SinglesAlphaCrAnSM[j], helper.SinglesAlphaCrAnSM[j + helper.size_single_beta]);
            }
            for (size_t j = helper.DoublesFromAlphaOffset[a]; j < helper.DoublesFromAlphaOffset[a + 1]; j++) {
                size_t ja = helper.DoublesFromAlphaKetIndex[j];
                size_t jb = ib;
                size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb - helper.ketBetaStart;
                ElemT WeightJ = this->T[ketIdx];
                this->TwoDiffCorrelation(DetI, WeightI, WeightJ,
                                    helper.DoublesAlphaCrAnSM[j], helper.DoublesAlphaCrAnSM[j + helper.size_double_beta],
                                    helper.DoublesAlphaCrAnSM[j + 2 * helper.size_double_beta], helper.DoublesAlphaCrAnSM[j + 3 * helper.size_double_beta]);
            }
        }
    }
};

/**
    Function to evaluate the two-particle correlation functions
*/
template <typename ElemT>
void Correlation(const std::vector<ElemT> &W_in,
                    const size_t adet_comm_size,
                    const size_t bdet_comm_size,
                    MultDataThrust<ElemT> &data,
                    MPI_Comm h_comm,
                    MPI_Comm b_comm,
                    MPI_Comm t_comm,
                    std::vector<std::vector<ElemT>> &onebody_out,
                    std::vector<std::vector<ElemT>> &twobody_out)
{
    thrust::device_vector<ElemT> onebody(data.norbs * data.norbs * 2, ElemT(0.0));
    thrust::device_vector<ElemT> twobody(data.norbs * data.norbs * data.norbs * data.norbs * 4, ElemT(0.0));

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm, &mpi_rank_h);
    MPI_Comm_size(h_comm, &mpi_size_h);

    int mpi_size_b;
    MPI_Comm_size(b_comm, &mpi_size_b);
    int mpi_rank_b;
    MPI_Comm_rank(b_comm, &mpi_rank_b);
    int mpi_size_t;
    MPI_Comm_size(t_comm, &mpi_size_t);
    int mpi_rank_t;
    MPI_Comm_rank(t_comm, &mpi_rank_t);
    size_t braAlphaSize = 0;
    size_t braBetaSize = 0;
    if (data.helper.size() != 0) {
        braAlphaSize = data.helper[0].braAlphaEnd - data.helper[0].braAlphaStart;
        braBetaSize = data.helper[0].braBetaEnd - data.helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = data.adets.size();
    size_t bdet_min = 0;
    size_t bdet_max = data.bdets.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    thrust::device_vector<ElemT> T(max_det_size);
    thrust::device_vector<ElemT> R;
    thrust::device_vector<ElemT> W(W_in.size());
    thrust::copy_n(W_in.begin(), W_in.size(), W.begin());

    if (data.helper.size() != 0) {
        Mpi2dSlide(W, T, adet_comm_size, bdet_comm_size,
                    -data.helper[0].adetShift, -data.helper[0].bdetShift, b_comm);
    }

    size_t offset = 0;
    size_t size = 0;
    if (mpi_rank_t == 0) {
        if (data.use_precalculated_dets) {
            // precalculate DetI (if update needed)
            data.UpdateDet(0);

            offset = 0;
            size = braAlphaSize * braBetaSize;
            while (offset < size) {
                size_t num_threads = data.num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                CorrelationInit kernel(data.helper[0], data, W, T, onebody, twobody, offset);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto ci = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                offset += num_threads;
            }
        } else {
            offset = 0;
            size = braAlphaSize * braBetaSize;
            while (offset < size) {
                size_t num_threads = data.num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                CorrelationInitNoCache kernel(data.helper[0], data, W, T, onebody, twobody, offset);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto ci = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                offset += num_threads;
            }
        }
    }

    for (size_t task = 0; task < data.helper.size(); task++) {
        size_t ketAlphaSize = data.helper[task].ketAlphaEnd - data.helper[task].ketAlphaStart;
        size_t ketBetaSize = data.helper[task].ketBetaEnd - data.helper[task].ketBetaStart;

        if (data.use_precalculated_dets) {
            // precalculate DetI (if update needed)
            data.UpdateDet(task);

            if (data.helper[task].taskType == 2) { // beta range are same
                offset = 0;
                size = data.helper[task].size_single_alpha * braBetaSize;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationSingleAlpha kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }

                offset = 0;
                size = data.helper[task].size_double_alpha * braBetaSize;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationDoubleAlpha kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            }
            else if (data.helper[task].taskType == 1) {
                offset = 0;
                size = data.helper[task].size_single_beta * braAlphaSize;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationSingleBeta kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }

                offset = 0;
                size = data.helper[task].size_double_beta * braAlphaSize;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationDoubleBeta kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            } else {
                offset = 0;
                size = data.helper[task].size_single_alpha * data.helper[task].size_single_beta;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationAlphaBeta kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            } // if ( helper[task].taskType ==  )
        } else {    // no det cache
            if (data.helper[task].taskType == 2) {
                offset = 0;
                size = braAlphaSize * braBetaSize;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }
                    CorrelationTask2 kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            } else if(data.helper[task].taskType == 1) {
                offset = 0;
                size = braAlphaSize * braBetaSize;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationTask1 kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            } else {
                offset = 0;
                size = braAlphaSize * braBetaSize;
                while (offset < size) {
                    size_t num_threads = data.num_max_threads;
                    if (offset + num_threads > size) {
                        num_threads = size - offset;
                    }

                    CorrelationTask0 kernel(data.helper[task], data, W, T, onebody, twobody, offset);
                    kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                    auto ci = thrust::counting_iterator<size_t>(0);
                    thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                    offset += num_threads;
                }
            }
        }

        if (data.helper[task].taskType == 0 && task != data.helper.size() - 1) {
            int adetslide = data.helper[task].adetShift - data.helper[task + 1].adetShift;
            int bdetslide = data.helper[task].bdetShift - data.helper[task + 1].bdetShift;
            R.resize(T.size());
            R = T;
            Mpi2dSlide(R, T, adet_comm_size, bdet_comm_size, adetslide, bdetslide, b_comm);
        }
    } // end for(size_t task=0; task < helper.size(); task++)

    if (mpi_size_b > 1)
        MpiAllreduce(onebody, MPI_SUM, b_comm);
    if (mpi_size_t > 1)
        MpiAllreduce(onebody, MPI_SUM, t_comm);
    if (mpi_size_h > 1)
        MpiAllreduce(onebody, MPI_SUM, h_comm);

    if (mpi_size_b > 1)
        MpiAllreduce(twobody, MPI_SUM, b_comm);
    if (mpi_size_t > 1)
        MpiAllreduce(twobody, MPI_SUM, t_comm);
    if (mpi_size_h > 1)
        MpiAllreduce(twobody, MPI_SUM, h_comm);


    // copy out onebody, twobody
    onebody_out.resize(2);
    size = data.norbs * data.norbs;
    offset = 0;
    for(int s=0; s < 2; s++) {
        onebody_out[s].resize(size, ElemT(0.0));
        thrust::copy_n(onebody.begin() + offset, size, onebody_out[s].begin());
        offset += size;
    }

    twobody_out.resize(4);
    size = data.norbs * data.norbs * data.norbs * data.norbs;
    offset = 0;
    for(int s=0; s < 4; s++) {
        twobody_out[s].resize(size, ElemT(0.0));
        thrust::copy_n(twobody.begin() + offset, size, twobody_out[s].begin());
        offset += size;
    }
}

} // end namespace sbd

#endif // end if for #ifndef SBD_CHEMISTRY_PTMB_CORRELATION_H
