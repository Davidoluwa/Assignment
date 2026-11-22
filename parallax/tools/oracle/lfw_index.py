#!/usr/bin/env python3
"""Independent LFW index for checking src/data/lfw.c (test use only; run with python3 -I).

usage: lfw_index.py ROOT OUT.csv
Writes "index,id,name,num,relpath" exactly as plx_lfw_write_csv does: identities sorted by
byte-wise name, images by number, only canonical "<Name>_<NNNN>.jpg" files, identities
without images dropped.
"""
import os
import re
import sys


def main(root, out):
    names = sorted((n for n in os.listdir(root) if not n.startswith(".") and os.path.isdir(os.path.join(root, n))),
                   key=lambda n: n.encode())
    rows, ident = [], 0
    for name in names:
        pat = re.compile(re.escape(name) + r"_(\d+)\.jpg\Z")
        nums = []
        for f in os.listdir(os.path.join(root, name)):
            m = pat.match(f)
            if m and not f.startswith(".") and f == f"{name}_{int(m.group(1)):04d}.jpg":
                nums.append(int(m.group(1)))
        if not nums:
            continue
        for n in sorted(nums):
            rows.append(f"{len(rows)},{ident},{name},{n},{name}/{name}_{n:04d}.jpg")
        ident += 1
    with open(out, "w") as f:
        f.write("index,id,name,num,relpath\n")
        f.writelines(r + "\n" for r in rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
