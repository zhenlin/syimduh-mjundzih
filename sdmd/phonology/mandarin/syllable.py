from enum import Enum

class MandarinInitial(Enum):
    B = 'b'
    P = 'p'
    M = 'm'
    F = 'f'
    D = 'd'
    T = 't'
    N = 'n'
    L = 'l'
    G = 'g'
    K = 'k'
    H = 'h'
    J = 'j'
    Q = 'q'
    X = 'x'
    ZH = 'zh'
    CH = 'ch'
    SH = 'sh'
    R = 'r'
    Z = 'z'
    C = 'c'
    S = 's'
    _ = ''


class MandarinMedial(Enum):
    I = 'i'
    U = 'u'
    V = 'ü'
    _ = ''


class MandarinNucleus(Enum):
    A = 'a'
    E = 'e'
    EH = 'ê'
    O = 'o'
    _ = ''


class MandarinCoda(Enum):
    I = 'i'
    U = 'u'
    N = 'n'
    NG = 'ng'
    R = 'r'
    _ = ''


class MandarinTone(Enum):
    _0 = 0
    _1 = 1
    _2 = 2
    _3 = 3
    _4 = 4


class MandarinSyllable:
    initial: MandarinInitial
    medial: MandarinMedial
    nucleus: MandarinNucleus
    coda: MandarinCoda
    tone: MandarinTone

    def __init__(self, initial: MandarinInitial, medial: MandarinMedial, nucleus: MandarinNucleus, coda: MandarinCoda, tone: MandarinTone):
        self.initial = initial
        self.medial = medial
        self.nucleus = nucleus
        self.coda = coda
        self.tone = tone
    
    def __repr__(self):
        return f'MandarinSyllable({repr(self.initial)}, {repr(self.medial)}, {repr(self.nucleus)}, {repr(self.coda)}, {repr(self.tone)})'

    def __str__(self):
        return f'({self.initial.value}, {self.medial.value}, {self.nucleus.value}, {self.coda.value}, {self.tone.value})'
