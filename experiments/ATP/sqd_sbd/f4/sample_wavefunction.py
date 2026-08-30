"""Sample bitstrings from an ASCI wavefunction's |amplitude|^2 distribution.

Reads a MACIS .wf file, converts occupation strings to the Qiskit convention
used by the SQD pipeline, and produces a counts pickle identical in format to
what load_and_process_bitstrings() outputs.

Qiskit convention (after transform_bitstring): <beta_reversed><alpha_reversed>
where "reversed" means highest MO index first.

ASCI occupation string characters:
  '2' = doubly occupied (1 alpha + 1 beta)
  'u' = alpha only
  'd' = beta only
  '0' = empty
"""

import argparse
import pickle
import collections

import numpy as np


def parse_wf_file(wf_path):
    """Parse a MACIS .wf file.

    Returns:
        norb: number of spatial orbitals
        nalpha: number of alpha electrons
        nbeta: number of beta electrons
        amplitudes: list of floats
        occ_strings: list of strings (each of length norb)
    """
    amplitudes = []
    occ_strings = []

    with open(wf_path) as f:
        header = f.readline().split()
        ndets = int(header[0])
        norb = int(header[1])
        nalpha = int(header[2])
        nbeta = int(header[3])

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            amp = float(parts[0])
            # Occupation string may be padded beyond norb; take first norb chars
            occ = parts[1][:norb]
            amplitudes.append(amp)
            occ_strings.append(occ)

    print(f"Parsed {len(amplitudes)} determinants (header: {ndets})")
    print(f"  norb={norb}, nalpha={nalpha}, nbeta={nbeta}")
    return norb, nalpha, nbeta, amplitudes, occ_strings


def occ_to_bitstring(occ, norb):
    """Convert ASCI occupation string to Qiskit-convention bitstring.

    ASCI: lowest MO first, left to right.
    Qiskit SQD pipeline: <beta_reversed><alpha_reversed>
      where reversed = highest MO index first.
    """
    alpha_bits = []
    beta_bits = []
    for ch in occ:
        if ch == '2':
            alpha_bits.append('1')
            beta_bits.append('1')
        elif ch == 'u':
            alpha_bits.append('1')
            beta_bits.append('0')
        elif ch == 'd':
            alpha_bits.append('0')
            beta_bits.append('1')
        else:  # '0'
            alpha_bits.append('0')
            beta_bits.append('0')

    # Reverse both (highest MO index first)
    alpha_bits.reverse()
    beta_bits.reverse()

    # Concatenate: beta_reversed + alpha_reversed
    return ''.join(beta_bits + alpha_bits)


def main():
    parser = argparse.ArgumentParser(
        description="Sample bitstrings from ASCI wavefunction"
    )
    parser.add_argument("--wf_file", type=str, required=True,
                        help="Path to MACIS .wf file")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of bitstrings to sample (default: 50000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output pickle path (default: counts_<num_samples>.pkl)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"counts_{args.num_samples}.pkl"

    # Parse wavefunction
    norb, nalpha, nbeta, amplitudes, occ_strings = parse_wf_file(args.wf_file)

    # Convert all occupation strings to Qiskit-convention bitstrings
    print("Converting occupation strings to bitstrings...")
    bitstrings = []
    for occ in occ_strings:
        bitstrings.append(occ_to_bitstring(occ, norb))

    # Verify Hamming weights on a sample
    nqubits = 2 * norb
    for i in range(min(10, len(bitstrings))):
        bs = bitstrings[i]
        assert len(bs) == nqubits, f"Bitstring length {len(bs)} != {nqubits}"
        beta_part = bs[:norb]
        alpha_part = bs[norb:]
        hw_alpha = sum(int(b) for b in alpha_part)
        hw_beta = sum(int(b) for b in beta_part)
        assert hw_alpha == nalpha, f"Det {i}: alpha HW {hw_alpha} != {nalpha}"
        assert hw_beta == nbeta, f"Det {i}: beta HW {hw_beta} != {nbeta}"
    print(f"  Hamming weight check passed (nalpha={nalpha}, nbeta={nbeta})")

    # Build probability distribution from |amplitude|^2
    amps = np.array(amplitudes)
    probs = amps ** 2
    total_prob = probs.sum()
    print(f"  Sum of |amp|^2 = {total_prob:.10f} (should be ~1.0)")
    probs /= total_prob  # normalize

    # Sample
    print(f"Sampling {args.num_samples} bitstrings (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(bitstrings), size=args.num_samples, replace=True, p=probs)

    # Build counts dict
    counts = collections.Counter()
    for idx in indices:
        counts[bitstrings[idx]] += 1

    print(f"  {len(counts)} unique bitstrings sampled")
    print(f"  Total shots: {sum(counts.values())}")

    # Top 5 by count
    print("  Top 5 bitstrings:")
    for bs, cnt in counts.most_common(5):
        print(f"    {bs}: {cnt}")

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump(dict(counts), f)
    print(f"Saved counts to {args.output}")


if __name__ == "__main__":
    main()
