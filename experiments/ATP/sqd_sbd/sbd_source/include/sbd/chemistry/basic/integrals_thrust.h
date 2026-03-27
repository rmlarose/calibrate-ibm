/**
@file sbd/chemistry/basic/integrals_thrust.h
@brief Functions to handle integrals
*/
#ifndef SBD_CHEMISTRY_BASIC_INTEGRALS_THRUST_H
#define SBD_CHEMISTRY_BASIC_INTEGRALS_THRUST_H


namespace sbd {

template <typename ElemT>
class oneInt_Thrust
{
protected:
    ElemT* store;
    int norbs;
public:
    oneInt_Thrust() {}

    oneInt_Thrust(const thrust::device_vector<ElemT>& v, int n)
    {
        store = (ElemT*)thrust::raw_pointer_cast(v.data());
        norbs = n;
    }

    oneInt_Thrust(const oneInt_Thrust& other)
    {
        store = other.store;
        norbs = other.norbs;
    }

    inline __device__ __host__ ElemT Value(int i, int j) const
    {
        return store[i * norbs + j];
    }
};

template <typename ElemT>
class twoInt_Thrust
{
protected:
    ElemT* store;
    ElemT maxEntry;
    ElemT zero;
    int norbs;
    ElemT* DirectMat;
    ElemT* ExchangeMat;
public:
    twoInt_Thrust() : zero(0.0), maxEntry(100.0) {}

    twoInt_Thrust(const thrust::device_vector<ElemT>& v, int n, const thrust::device_vector<ElemT>& dm, const thrust::device_vector<ElemT>& em, ElemT z = 0.0, ElemT mx = 100.0)  : zero(z), maxEntry(mx)
    {
        store = (ElemT*)thrust::raw_pointer_cast(v.data());
        norbs = n;
        DirectMat = (ElemT*)thrust::raw_pointer_cast(dm.data());
        ExchangeMat = (ElemT*)thrust::raw_pointer_cast(em.data());
    }

    twoInt_Thrust(const twoInt_Thrust& other)
    {
        store = other.store;
        norbs = other.norbs;
        maxEntry = other.maxEntry;
        zero = other.zero;
        DirectMat = other.DirectMat;
        ExchangeMat = other.ExchangeMat;
    }

    inline __device__ __host__ ElemT &Direct(int i, int j)
    {
        return DirectMat[i + norbs * j];
    }
    inline __device__ __host__ ElemT &Exchange(int i, int j)
    {
        return ExchangeMat[i + norbs * j];
    }

    inline __device__ __host__ ElemT Value(int i, int j, int k, int l) const
    {
        if (!((i % 2 == j % 2) && (k % 2 == l % 2)))
            return zero;
        int I = i / 2;
        int J = j / 2;
        int K = k / 2;
        int L = l / 2;
        int ij = std::max(I, J) * (std::max(I, J) + 1) / 2 + std::min(I, J);
        int kl = std::max(K, L) * (std::max(K, L) + 1) / 2 + std::min(K, L);
        int a = std::max(ij, kl);
        int b = std::min(ij, kl);
        return store[a * (a + 1) / 2 + b];
    }
    inline __device__ __host__ ElemT DirectValue(int i, int j) const
    {
        return DirectMat[i + norbs * j];
    }
    inline __device__ __host__ ElemT ExchangeValue(int i, int j) const
    {
        return ExchangeMat[i + norbs * j];
    }
};

} // end namespace sbd

#endif // end SBD_CHEMISTRY_INTEGRALS_H
