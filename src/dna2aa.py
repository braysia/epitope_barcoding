from __future__ import division
from __future__ import print_function

import re
import numpy as np

_CODON2LETTER = {
    'TAA': '.',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'RAC': 'B', 'RAT': 'B',
    'TGT': 'C', 'TGC': 'C',
    'GAT': 'D', 'GAC': 'D',
    'GAG': 'E', 'GAA': 'E',
    'TTT': 'F', 'TTC': 'F',
    'GGT': 'G', 'GGG': 'G', 'GGA': 'G', 'GGC': 'G',
    'CAT': 'H', 'CAC': 'H',
    'ATC': 'I', 'ATA': 'I', 'ATT': 'I',
    'MTT': 'J', 'MTC': 'J', 'MTA': 'J',
    'AAG': 'K', 'AAA': 'K',
    'CTT': 'L', 'CTG': 'L', 'CTA': 'L', 'CTC': 'L', 'TTA': 'L', 'TTG': 'L',
    'ATG': 'M',
    'AAC': 'N', 'AAT': 'N',
    'TAG': 'O',
    'CCT': 'P', 'CCG': 'P', 'CCA': 'P', 'CCC': 'P',
    'CAA': 'Q', 'CAG': 'Q',
    'AGG': 'R', 'AGA': 'R', 'CGA': 'R', 'CGG': 'R', 'CGT': 'R', 'CGC': 'R',
    'AGC': 'S', 'AGT': 'S', 'TCT': 'S', 'TCG': 'S', 'TCC': 'S', 'TCA': 'S',
    'ACA': 'T', 'ACG': 'T', 'ACT': 'T', 'ACC': 'T',
    'TGA': 'U',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'TGG': 'W',
    'NNN': 'X',
    'TAT': 'Y', 'TAC': 'Y',
    'SAA': 'Z', 'SAG': 'Z',
}


def codon2letter(codon):
    """Convert codon into alphabet.
    Parameters
    ----------
    codon : str
        Codon.
    Returns
    -------
    str
        Alphabetical expression of the given code. When the input value is not
        valid codon (three alphabets either all upper case or all lower case),
        no conversion is carried out and the input value is returned as it is.
    """
    if re.match(r'[a-z]{3}', codon):
        return _CODON2LETTER[codon.upper()].lower()
    if re.match(r'[A-Z]{3}', codon):
        return _CODON2LETTER[codon]
    return codon


def dna2aa(dna):
    """Convert DNA sequence into text.
    Parameters
    ----------
    dna : str
        Input DNA sequence. Must be ASCII string. Can contain punctuations.
    Returns
    -------
    str
        Text representation of the given DNA sequence.
    """
    splitted = re.findall(r'(?:[^a-zA-Z]|[a-zA-Z]{3})', dna)
    return ''.join(codon2letter(i) for i in splitted)