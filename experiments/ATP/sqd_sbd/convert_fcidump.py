"""Convert text FCIDUMP to binary format for fast SBD loading.

Binary layout: [int32 NORB][int32 NELEC][int64 N][N × {float64 val, int32 i,j,k,l}]
Each record is 24 bytes. Single fread() instead of per-line stringstream parsing.
"""

import struct
import sys
import os
import numpy as np


def convert_fcidump(input_path, output_path):
    norb = nelec = None
    values = []
    indices_i = []
    indices_j = []
    indices_k = []
    indices_l = []
    in_header = True

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            if in_header:
                if '&END' in line:
                    in_header = False
                    continue
                clean = line.replace('&FCI', '')
                for part in clean.split(','):
                    part = part.strip()
                    if '=' in part:
                        key, val = part.split('=', 1)
                        key = key.strip()
                        val = val.strip()
                        if key == 'NORB':
                            norb = int(val)
                        elif key == 'NELEC':
                            nelec = int(val)
            else:
                parts = line.split()
                values.append(float(parts[0]))
                indices_i.append(int(parts[1]))
                indices_j.append(int(parts[2]))
                indices_k.append(int(parts[3]))
                indices_l.append(int(parts[4]))

    n = len(values)

    # Build structured array matching C++ FCIDumpRecord layout
    dt = np.dtype([('value', '<f8'), ('i', '<i4'), ('j', '<i4'),
                   ('k', '<i4'), ('l', '<i4')])
    assert dt.itemsize == 24, f"Record size mismatch: {dt.itemsize} != 24"

    records = np.empty(n, dtype=dt)
    records['value'] = values
    records['i'] = indices_i
    records['j'] = indices_j
    records['k'] = indices_k
    records['l'] = indices_l

    with open(output_path, 'wb') as f:
        f.write(struct.pack('iiq', norb, nelec, n))  # 16-byte header
        f.write(records.tobytes())

    text_size = os.path.getsize(input_path)
    bin_size = os.path.getsize(output_path)
    print(f"Converted: {input_path}")
    print(f"  -> {output_path}")
    print(f"  NORB={norb}, NELEC={nelec}, {n} integrals")
    print(f"  {text_size:,} bytes text -> {bin_size:,} bytes binary "
          f"({text_size/bin_size:.1f}x smaller)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_fcidump.py input.fcidump [output.fcidump.bin]")
        sys.exit(1)

    input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = input_path + '.bin'

    convert_fcidump(input_path, output_path)
