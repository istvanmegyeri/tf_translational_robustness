import numpy as np


# DeepSEA:  1,000 × 4 binary matrix, with columns corresponding to A, G, C and T.
ohe_nuc_d = {
    'A':[True, False, False, False],#[1,0,0,0],
    'G':[False, True, False, False],#[0,1,0,0],
    'C':[False, False, True, False],#[0,0,1,0],
    'T':[False, False, False, True],#[0,0,0,1]
    'N':[False, False, False, False],#[0,0,0,0]
}


def ohe2nuc1(ohe):
    nl = []  # Nukleotid Lista
    for i, cn in enumerate(ohe):
        if list(cn) == ohe_nuc_d['A']:
            nl.append('A')
        elif list(cn) == ohe_nuc_d['C']:
            nl.append('C')
        elif list(cn) == ohe_nuc_d['G']:
            nl.append('G')
        elif list(cn) == ohe_nuc_d['T']:
            nl.append('T')
        elif list(cn) == ohe_nuc_d['N']:
            nl.append('N')
    return "".join(nl)


def nuc2ohe1(nuc):
    nl = []  # Nukleotid Lista
    for idx_n2o, cn in enumerate(nuc):
        if cn == 'A':
            nl.append(ohe_nuc_d['A'])
        elif cn == 'C':
            nl.append(ohe_nuc_d['C'])
        elif cn == 'G':
            nl.append(ohe_nuc_d['G'])
        elif cn == 'T':
            nl.append(ohe_nuc_d['T'])
        elif cn == 'N':
            nl.append(ohe_nuc_d['N'])
    return nl


"""
Usage example
"""

ex_seq = "ACGTTTGGCCAA"

# Conversion between nuc and ohe
ex_seq_ohe = nuc2ohe1(ex_seq)
ex_seq_nuc = ohe2nuc1(ex_seq_ohe)

# Getting Reverse complement

print(ohe2nuc1(ex_seq_ohe))
print(ohe2nuc1(np.flip(ex_seq_ohe))[::-1])