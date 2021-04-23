import numpy as np
import pysam

import sys
from dna2aa import dna2aa


P2A = 'GGAAGCGGAGCTACTAACTTCAGCCTGCTGAAGCAGGCTGGAGACGTGGAGGAGAACCCTGGACCT'.upper()
BINI = 'gggcgcgcc'.upper()  # linker sequences immediately before the start of barcodes
NINI = 'AGgccgcacataGGTACCga'.upper()  # sequences immediately before the start of N8
spacer = 3  # To account for short N8. Not required if NNN oligo is high quality (PAGE purified)
BINI = BINI[:-spacer]


tag_dict = dict(flag='DYKDDDDK', ha='YPYDVPDYA', his='HHHHHH', myc='EQKLISEEDL', v5='IPNPLLGLD',
                au5='TDFYLK', t7='MASMTGGQQMG', tag100='EETARFQPGYRS', strep='SAWSHPQFEK',
                nws='NWSHPQFEK', univ='HTTPHH', vsv='YTDIEMNRLGK', proc='EDQVDPRLIDGK', etag='APVPYPDPLEPR',
                srt='TFIGAIATDT', mat='HNHRHKH', glu='CEEEEYMPME', oll='FANELGPRLMGK')
short_tag_dict = {}
for k, v in tag_dict.items():
    short_tag_dict[k] = v[:3]


def sampair2list(sf):
    rset0, rset1 = [], []
    for read in sf:
        if read.is_read1:
            read0 = read
            qname = read0.query_name
            continue
        else:
            read1 = read
            if read1.query_name == qname:
                rset0.append(read0)
                rset1.append(read1)
    return rset0, rset1


def read_iscontain(rset0, rset1):
    """Check if a read contains the functional region"""
    fset0, fset1 = [], []
    for r0, r1 in zip(rset0, rset1):
        if P2A.upper() in r0.tostring() and BINI.upper() in r0.tostring() and NINI in r1.tostring():
            fset0.append(r0)
            fset1.append(r1)
    return fset0, fset1


def translateandmatch(s):
    """translate(seq[gra_idx:])"""
    aa = dna2aa(s)
    st = []
    for k, v in tag_dict.items():
        idx = aa.find(v)
        if idx > 0:
            if idx < 38:
                st.append(k)
            else:
                st.append("".join(['N', k]))
        if idx > 83:
            pass
    for k, v in short_tag_dict.items():
        idx1 = aa[83:86].find(v)
        if idx1 > 0:
            st.append("".join(['N', k]))
    return st


def translateandmatch(s):
    """translate(seq[BINI:])
    To account for the quality decay as it approaches to 300nt, 
    we only check 3 aa for the epitope in the last cassette.
    """
    aa = dna2aa(s)
    st = []
    for k, v in tag_dict.items():
        idx = aa.find(v)
        if idx > 0:
            if idx < 38:  # corresponding to the barcode fused with TOMM20
                st.append(k)
            elif idx >= 83:  # before the last cassette
                pass
            else:  # corresponding to the barcode fused with H2B
                st.append("".join(['N', k]))
    for k, v in short_tag_dict.items():
        idx1 = aa[83:86].find(v)
        if idx1 >= 0:
            st.append("".join(['N', k]))
    return st


def calc_store(fset0, fset1, qt=25):
    store = []
    for f0, f1 in zip(fset0, fset1):
        seq0 = f0.tostring().split('\t')[9]
        gra_idx = seq0.find(BINI)
        epis = translateandmatch(seq0[gra_idx:])
        seq1 = f1.tostring().split('\t')[9]
        qua1 = f1.tostring().split('\t')[10]
        if NINI.upper() in seq1:
            nstart = seq1.find(NINI) + len(NINI) + spacer
            nnns = seq1[nstart:nstart+9]
            if np.min(f1.query_qualities[nstart:nstart+9]) < qt:
                continue
            store.append([nnns, ] + epis)
    return store


def filter_store(store, min_reads=5, min_rep=0.4):
    """
    - For each NNNs, it needs to have at least min_reads reads with same set of epitopes.
      (this may be little aggressive for a sample data)
    - To account for template switching during PCR and sequencing, discard the set 
      if it's representation is less than min_rep out of reads posessing the same NNNs.
    """
    store1 = []
    unnns = np.unique([i[0] for i in store])
    for unn in unnns:
        pst = [i for i in store if i[0]==unn and len(i)>1]
        if not pst or len(pst) < min_reads:
            continue
        upst = [list(x) for x in set(tuple(x) for x in pst)]
        counts = [pst.count(u) for u in upst]
        idx = np.argmax(counts)
        if np.max(counts)/np.sum(counts) > min_rep:
            store1.append(upst[idx])
    return store1


if __name__ == "__main__":
    """the following saves a list containing N8 sequences and the corresponding set of epitopes"""
    path0 = 'data/epi2n.sam'
    samfile = pysam.AlignmentFile(path0, "rb", check_sq=False)
    sf = samfile.fetch()
    rset0, rset1 = sampair2list(sf)
    fset0, fset1 = read_iscontain(rset0, rset1)
    store = calc_store(fset0, fset1)
    store1 = filter_store(store)
    np.savez('nandbar.npz', store1)

