# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import numpy as np
from numbers import Number

from .masses import U_MASS_EV
from .masses import __dict__ as mass__dict__

# Monte Carlo numbering scheme as defined by the Particle Data Group
# See https://pdg.lbl.gov/2007/reviews/montecarlorpp.pdf for implementation
# details.
# Not all particles are implemented yet; this can be appended later


# This is the internal dictionary of PDG IDs.
# For each pdg id, we get the charge and a list of names.
# The first name is the default short name, the last is the default long name.
# The other names are accepted alternatives. Any superscript or subscript can be used
# interchangeably with its normal script (if both are used, subscript comes first).

pdg_table = {
#   ID       q  NAME
    0:     [0,  'undefined'],
    11:    [-1, 'e⁻', 'e', 'electron'],
    -11:   [1,  'e⁺', 'positron'],
    12:    [0,  'νₑ', 'electron neutrino'],
    13:    [-1, 'μ⁻', 'μ', 'muon-', 'muon'],
    -13:   [1,  'μ⁺', 'muon+', 'anti-muon'],
    14:    [0,  'νμ', 'muon neutrino'],
    15:    [-1, 'τ⁻', 'τ', 'tau-', 'tau'],
    -15:   [1, 'τ⁺', 'tau+', 'anti-tau'],
    16:    [0,  'ντ', 'tau neutrino'],
    22:    [0,  'γ⁰', 'γ', 'gamma', 'photon'],
    111:   [0,  'π⁰', 'π', 'pion', 'pion0', 'pi0'],
    211:   [1,  'π⁺', 'pion+', 'pi+'],
    -211:  [-1, 'π⁻', 'pion-', 'pi-'],
    311:   [0,  'K⁰', 'kaon', 'kaon0'],
    321:   [1,  'K⁺', 'kaon+'],
    -321:  [-1, 'K⁻', 'kaon-'],
    130:   [0,  'KL', 'long kaon'],
    310:   [0,  'Kₛ', 'short kaon'],
    421:   [0,  'D⁰', 'D'],
    411:   [1,  'D⁺'],
    -411:  [-1, 'D⁻'],
    431:   [1,  'Dₛ⁺'],
    -431:  [-1, 'Dₛ⁻'],
    2212:  [1,  'p⁺', 'p', 'proton'],
    -2212: [-1,  'p⁻', 'anti-proton'],
    2112:  [0,  'n⁰', 'n', 'neutron'],
    2224:  [2,  'Δ⁺⁺', 'delta++'],
    2214:  [1,  'Δ⁺', 'delta+'],
    2114:  [0,  'Δ⁰', 'delta0'],
    1114:  [-1, 'Δ⁻', 'delta-'],
    3122:  [0,  'Λ⁰', 'Λ', 'lambda'],
    4122:  [1,  'Λc⁺', 'lambdac+'],
    3222:  [1,  'Σ⁺', 'sigma+'],
    3212:  [0,  'Σ⁰', 'Σ', 'sigma', 'sigma0'],
    3112:  [-1, 'Σ⁻', 'sigma-'],
    3322:  [0,  'Ξ⁰', 'Ξ', 'xi', 'xi0'],
    3312:  [-1, 'Ξ⁻', 'xi-'],
    4132:  [0,  'Ξc⁰', 'Ξc', 'xic', 'xic0'],
    4232:  [1,  'Ξc⁺', 'xic+'],
    4312:  [0,  "Ξ'c⁰", "Ξ'c", "xiprimec", "xiprimec0"],
    4322:  [1,  "Ξ'c⁺", "xiprimec+"],
    3334:  [-1, 'Ω⁻', 'omega-'],
    4332:  [0, 'Ωc⁰', 'Ωc', 'omegac', 'omegac0'],
    1000010020: [1, '²H', 'H2', 'hydrogen-2', 'deuteron'],
    1000010030: [1, '³H', 'H3', 'hydrogen-3', 'triton']
}

elements = {
     1: "H",    2: "He",   3: "Li",   4: "Be",   5: "B",    6: "C",    7: "N",    8: "O",    9: "F",   10: "Ne",
    11: "Na",  12: "Mg",  13: "Al",  14: "Si",  15: "P",   16: "S",   17: "Cl",  18: "Ar",  19: "K",   20: "Ca",
    21: "Sc",  22: "Ti",  23: "V",   24: "Cr",  25: "Mn",  26: "Fe",  27: "Co",  28: "Ni",  29: "Cu",  30: "Zn",
    31: "Ga",  32: "Ge",  33: "As",  34: "Se",  35: "Br",  36: "Kr",  37: "Rb",  38: "Sr",  39: "Y",   40: "Zr",
    41: "Nb",  42: "Mo",  43: "Tc",  44: "Ru",  45: "Rh",  46: "Pd",  47: "Ag",  48: "Cd",  49: "In",  50: "Sn",
    51: "Sb",  52: "Te",  53: "I",   54: "Xe",  55: "Cs",  56: "Ba",  57: "La",  58: "Ce",  59: "Pr",  60: "Nd",
    61: "Pm",  62: "Sm",  63: "Eu",  64: "Gd",  65: "Tb",  66: "Dy",  67: "Ho",  68: "Er",  69: "Tm",  70: "Yb",
    71: "Lu",  72: "Hf",  73: "Ta",  74: "W",   75: "Re",  76: "Os",  77: "Ir",  78: "Pt",  79: "Au",  80: "Hg",
    81: "Tl",  82: "Pb",  83: "Bi",  84: "Po",  85: "At",  86: "Rn",  87: "Fr",  88: "Ra",  89: "Ac",  90: "Th",
    91: "Pa",  92: "U",   93: "Np",  94: "Pu",  95: "Am",  96: "Cm",  97: "Bk",  98: "Cf",  99: "Es", 100: "Fm",
   101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
   111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}
_elements_inv = {vv: kk for kk, vv in elements.items()}

elements_long = {
    1: "Hydrogen",        2: "Helium",          3: "Lithium",         4: "Beryllium",       5: "Boron",
    6: "Carbon",          7: "Nitrogen",        8: "Oxygen",          9: "Fluorine",       10: "Neon",
   11: "Sodium",         12: "Magnesium",      13: "Aluminum",       14: "Silicon",        15: "Phosphorus",
   16: "Sulfur",         17: "Chlorine",       18: "Argon",          19: "Potassium",      20: "Calcium",
   21: "Scandium",       22: "Titanium",       23: "Vanadium",       24: "Chromium",       25: "Manganese",
   26: "Iron",           27: "Cobalt",         28: "Nickel",         29: "Copper",         30: "Zinc",
   31: "Gallium",        32: "Germanium",      33: "Arsenic",        34: "Selenium",       35: "Bromine",
   36: "Krypton",        37: "Rubidium",       38: "Strontium",      39: "Yttrium",        40: "Zirconium",
   41: "Niobium",        42: "Molybdenum",     43: "Technetium",     44: "Ruthenium",      45: "Rhodium",
   46: "Palladium",      47: "Silver",         48: "Cadmium",        49: "Indium",         50: "Tin",
   51: "Antimony",       52: "Tellurium",      53: "Iodine",         54: "Xenon",          55: "Cesium",
   56: "Barium",         57: "Lanthanum",      58: "Cerium",         59: "Praseodymium",   60: "Neodymium",
   61: "Promethium",     62: "Samarium",       63: "Europium",       64: "Gadolinium",     65: "Terbium",
   66: "Dysprosium",     67: "Holmium",        68: "Erbium",         69: "Thulium",        70: "Ytterbium",
   71: "Lutetium",       72: "Hafnium",        73: "Tantalum",       74: "Tungsten",       75: "Rhenium",
   76: "Osmium",         77: "Iridium",        78: "Platinum",       79: "Gold",           80: "Mercury",
   81: "Thallium",       82: "Lead",           83: "Bismuth",        84: "Polonium",       85: "Astatine",
   86: "Radon",          87: "Francium",       88: "Radium",         89: "Actinium",       90: "Thorium",
   91: "Protactinium",   92: "Uranium",        93: "Neptunium",      94: "Plutonium",      95: "Americium",
   96: "Curium",         97: "Berkelium",      98: "Californium",    99: "Einsteinium",   100: "Fermium",
  101: "Mendelevium",   102: "Nobelium",      103: "Lawrencium",    104: "Rutherfordium", 105: "Dubnium",
  106: "Seaborgium",    107: "Bohrium",       108: "Hassium",       109: "Meitnerium",    110: "Darmstadtium",
  111: "Roentgenium",   112: "Copernicium",   113: "Nihonium",      114: "Flerovium",     115: "Moscovium",
  116: "Livermorium",   117: "Tennessine",    118: "Oganesson"
}
_elements_long_inv = {vv: kk for kk, vv in elements_long.items()}


def is_proton(pdg_id):
    """Check if a PDG ID corresponds to a proton."""
    return int(pdg_id) == 2212

def is_ion(pdg_id):
    """Check if a PDG ID corresponds to a heavy ion (A+Z > 1)."""
    tmpid = pdg_id - 1000000000
    L = int(tmpid/1e7)
    tmpid -= L*1e7
    Z = int(tmpid /1e4)
    tmpid -= Z*1e4
    A = int(tmpid /10)
    return (int(pdg_id) >= 1000000000) and (Z > 0) and (A > Z)

def is_lepton(pdg_id):
    """Check if a PDG ID corresponds to a lepton (neutrinos included)."""
    return 11 <= abs(int(pdg_id)) <= 16


def get_name_from_pdg_id(pdg_id, long_name=True, subscripts=True):
    """
    Get the name of a particle from its PDG ID.

    Parameters
    ----------
    long_name : bool, default True
        If True, return the long name of the particle (ASCII-compliant). For a
        particle in the PDF internal table, this is the last name in the list
        of alternatives. For an ion, it is the full element name, followed by
        the mass number. If 'long_name' is False, a short name is returned
        (Unicode). For a particle in the PDF internal table, this is the first
        name in the list of alternatives. For an ion, it is the element name
        preceded (when 'subscripts' is True) of followed (when 'subscripts' is
        False) by the mass number.
    subscripts : bool, default True
        Controls whether or not to allow sub- and superscripts in the short
        name. Has no function if 'long_name' is True.

    Returns
    -------
    str
        The name of the particle.
    """
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        return np.array([get_name_from_pdg_id(pdg) for pdg in pdg_id])
    return get_properties_from_pdg_id(pdg_id, long_name=long_name, subscripts=subscripts)[-1]


def get_pdg_id_from_name(name=None):
    """
    Get a particle's PDG ID from its name.

    Parameters
    ----------
    name : str
        The name of the particle. Can be any alternative from the PDF internal
        table, with or without sub- or superscripts. For ions, the name can be
        any combination of the element short or long name and the mass number,
        for instance 'Pb208', 'Pb 208', Pb-208', 'Pb_208', '208Pb', 'Lead-208',
        'lead 208', etc.

    Returns
    -------
    int
        The PDG ID of the particle.
    """
    if hasattr(name, 'get'):
        name = name.get()

    if name is None:
        return 0  # undefined
    elif hasattr(name, '__len__') and not isinstance(name, str):
        return np.array([get_pdg_id_from_name(nn) for nn in name])
    elif isinstance(name, Number):
        return int(name) # fallback

    lname = _to_normal_script(name).lower().replace('ς', 'σ')
    aname = ""
    if len(lname) > 4 and lname[:5]=="anti-":
        aname = _flip_end_sign(lname[5:])
    elif len(lname) > 4 and lname[:5]=="anti ":
        aname = _flip_end_sign(lname[5:])
    elif len(lname) > 3 and lname[:4]=="anti":
        aname = _flip_end_sign(lname[4:])

    _PDG_inv = get_pdg_inv()

    # particle
    if lname in _PDG_inv.keys():
        return _PDG_inv[lname]

    # anti-particle
    elif aname in _PDG_inv.keys():
        return -_PDG_inv[aname]

    else:
        ion_name = lname.lower()
        ion_name = ion_name.replace('_','').replace('-','')
        ion_name = ion_name.replace(' ','').replace('.','')
        for Z, ion in elements_long.items():
            if ion.lower() in ion_name:
                A = ion_name.replace(ion.lower(), '')
                if A.isnumeric() and int(A) > 0:
                    return get_pdg_id_ion(int(A), Z)
        for Z, ion in elements.items():
            if ion.lower() in ion_name:
                A = ion_name.replace(ion.lower(), '')
                if A.isnumeric() and int(A) > 0:
                    return get_pdg_id_ion(int(A), Z)
        raise ValueError(f"Particle {name} not found in pdg dictionary, or wrongly "
                        + f"formatted ion name!\nFor ions, use e.g. 'Pb208', 'Pb 208', "
                        + f"'Pb-208', 'Pb_208', 'Pb.208', '208Pb', 'lead-208', ...")


def get_properties_from_pdg_id(pdg_id, long_name=False, subscripts=True):
    """
    Get the properties of a particle from its PDG ID.

    Parameters
    ----------
    pdg_id : int or str
        The PDG ID of the particle.
    long_name : bool, default True
        If True, return the long name of the particle (ASCII-compliant). For a
        particle in the PDF internal table, this is the last name in the list
        of alternatives. For an ion, it is the full element name, followed by
        the mass number. If 'long_name' is False, a short name is returned
        (Unicode). For a particle in the PDF internal table, this is the first
        name in the list of alternatives. For an ion, it is the element name
        preceded (when 'subscripts' is True) of followed (when 'subscripts' is
        False) by the mass number.
    subscripts : bool, default True
        Controls whether or not to allow sub- and superscripts in the short
        name. Has no function if 'long_name' is True.
    name : str

    Returns
    -------
    q : int
        The charge of the particle.
    A : int
        The mass number of the particle (total number of protons and neutrons).
    Z : int
        The atomic number of the particle (number of protons).
    name : str
        The name of the particle, with formatting depending on 'long_name' and
        'subscripts'.
    """
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        result = np.array([get_properties_from_pdg_id(pdg) for pdg in pdg_id]).T
        return result[0].astype(np.float64), result[1].astype(np.int64),\
               result[2].astype(np.int64), result[3]

    pdg_id = int(pdg_id)

    if pdg_id in pdg_table.keys():
        q = pdg_table[pdg_id][0]
        if long_name:
            name = pdg_table[pdg_id][-1]
        else:
            name = pdg_table[pdg_id][1]
            if not subscripts:
                name = _to_normal_script(name)
        if abs(pdg_id)==2212 or abs(pdg_id)==2112:
            A = 1
            Z = q
        elif pdg_id==1000010020:
            A = 2
            Z = 1
            if not long_name and not subscripts:
                name = 'H2'
        elif pdg_id==1000010030:
            A = 3
            Z = 1
            if not long_name and not subscripts:
                name = 'H3'
        else:
            A = 0
            Z = 0
        return int(q), int(A), int(Z), name

    elif pdg_id > 1000000000:
        # Ion
        tmpid = pdg_id - 1000000000
        L = int(tmpid/1e7)
        tmpid -= L*1e7
        Z = int(tmpid /1e4)
        tmpid -= Z*1e4
        A = int(tmpid /10)
        tmpid -= A*10
        isomer_level = int(tmpid)
        if long_name:
            return int(Z), int(A), int(Z), f'{get_element_full_name_from_Z(Z)}-{A}'
        else:
            if subscripts:
                name = f'{_digits_to_superscript(A)}{get_element_name_from_Z(Z)}'
            else:
                name = f'{get_element_name_from_Z(Z)}{A}'
            return int(Z), int(A), int(Z), name

    elif -pdg_id in pdg_table.keys() or pdg_id < -1000000000:
        antipart = get_properties_from_pdg_id(-pdg_id, long_name=long_name, subscripts=subscripts)
        name = _flip_end_sign(f'anti-{antipart[3]}')
        return -antipart[0], antipart[1], -antipart[2], name

    else:
        raise ValueError(f"PDG ID {pdg_id} not recognised!")


def get_element_name_from_Z(z):
    """Get the short element name for the element with Z protons."""
    if z not in elements:
        raise ValueError(f"Element with {z} protons not known.")
    return elements[z]

def get_element_full_name_from_Z(z):
    """Get the full element name for the element with Z protons."""
    if z not in elements_long:
        raise ValueError(f"Element with {z} protons not known.")
    return elements_long[z]

def get_Z_from_element_name(name):
    """Get the atomic number from the short or long element name."""
    if name in _elements_inv:
        return _elements_inv[name]
    elif name in _elements_long_inv:
        return _elements_long_inv[name]
    else:
        raise ValueError(f"Element {name} not recognised!")


def get_pdg_id_ion(A, Z):
    """Get the PDG ID for an ion with Z protons and A-Z neutrons."""
    if hasattr(A, '__len__') and not isinstance(A, str) \
    and hasattr(Z, '__len__') and not isinstance(Z, str):
        return np.array([get_pdg_id_ion(aa, zz) for aa, zz in zip(A,Z)])
    Z = int(Z)
    A = int(A)
    if Z < 1 or A < 2 or A < Z:
        raise ValueError(f"Not an ion ({Z=} and {A=})!")
    return 1000000000 + Z*10000 + A*10


def get_pdg_id_from_mass_charge(m, q, rtol=1e-3):
    """Get the PDG ID for a given mass and charge. This is always an estimate."""
    if hasattr(q, '__len__') and not isinstance(q, str) \
    and hasattr(m, '__len__') and not isinstance(m, str):
        return np.array([get_pdg_id_from_mass_charge(mm, qq) for qq, mm in zip(q, m)])
    elif hasattr(q, '__len__') and not isinstance(q, str):
        return np.array([get_pdg_id_from_mass_charge(m, qq) for qq in q])
    elif hasattr(m, '__len__') and not isinstance(m, str):
        return np.array([get_pdg_id_from_mass_charge(mm, q) for mm in m])

    m = float(m)
    q = int(q)
    A = round(m/U_MASS_EV)

    # First we check the internal table of masses
    found_ids = []
    for pdg_id, val in _get_mass_table().items():
        if abs(val) < 1e-12:
            if abs(m) < 1e-12:
                found_ids.append(pdg_id)
        elif abs(val-m)/val <= rtol:
            found_ids.append(pdg_id)
    if len(found_ids) > 0:
        pdg_id = []
        for this_pdg_id in found_ids:
            this_q, _, _, _ = get_properties_from_pdg_id(this_pdg_id)
            if this_q == q:
                pdg_id.append(this_pdg_id)
            else:
                this_q, _, _, _ = get_properties_from_pdg_id(-this_pdg_id)
                if this_q == q:
                    pdg_id.append(-this_pdg_id)
        if len(pdg_id) == 1:
            return pdg_id[0]
        elif len(pdg_id) > 1:
            raise ValueError(f"Multiple particles found for {m=} and {q=}: {found_ids}. "
                           + f"Decrease the tolerance. If this does not work, it might be "
                           + f"ambiguous and not resolvable with mass and charge alone "
                           + f"(like for a neutral particle that is not its own anti-particle).")

    # Then we check for ions
    if q > 0 and A > 1:
        return get_pdg_id_ion(A, q)

    # Not found
    else:
        raise ValueError(f"Particle with charge {q} and mass {m} eV not recognised!\nIf "
                       + f"the particle mass is expected to be present in the internal "
                       + f"mass table in Xtrack, try increasing the tolerance.")


def get_mass_from_pdg_id(pdg_id, allow_approximation=True, expected_mass=None, verbose=True):
    """Get the particle mass for a given PDG ID, if in the internal database. If not, it can be extrapolated for ions."""
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        return np.array([get_mass_from_pdg_id(pdg,
                                      allow_approximation=allow_approximation,
                                      expected_mass=expected_mass)
                         for pdg in pdg_id])

    pdg_id = int(pdg_id)
    _mass_table = _get_mass_table()
    if pdg_id in _mass_table:
        return _mass_table[pdg_id]
    elif -pdg_id in _mass_table:
        return _mass_table[-pdg_id]
    elif pdg_id > 1000000000 and allow_approximation:
        _, A, _, _ = get_properties_from_pdg_id(pdg_id)
        if verbose:
            print(f"Warning: approximating the mass as {A}u!")
        return A*U_MASS_EV
    elif expected_mass is not None and _mass_consistent(pdg_id, expected_mass):
        # This is a workaround in case an exact mass is given
        # (like for the reference particle)
        return expected_mass
    else:
        raise ValueError(f"Exact mass not found for particle with PDG ID {pdg_id}.")


# Dynamically set _pdg_inv so it can be defined only here
_PDG_inv = {}
def get_pdg_inv():
    if _PDG_inv == {}:
        for pdg_id, val in pdg_table.items():
            for vv in val[1:]:
                _PDG_inv[_to_normal_script(vv).lower()] = pdg_id
    return _PDG_inv


def _flip_end_sign(name):
    if name[-2:] == '⁺⁺':
        return name[:-2] + '⁻⁻'
    elif name[-2:] == '++':
        return name[:-2] + '--'
    elif name[-2:] == '⁻⁻':
        return name[:-2] + '⁺⁺'
    elif name[-2:] == '--':
        return name[:-2] + '++'
    elif name[-1] == '⁺':
        return name[:-1] + '⁻'
    elif name[-1] == '+':
        return name[:-1] + '-'
    elif name[-1] == '⁻':
        return name[:-1] + '⁺'
    elif name[-1] == '-':
        return name[:-1] + '+'
    else:
        return name


def _digits_to_superscript(val):
    val = f'{val}'.replace('0', '⁰').replace('1', '¹').replace('2', '²').replace('3', '³')
    val = val.replace('4', '⁴').replace('5', '⁵').replace('6', '⁶').replace('7', '⁷')
    return val.replace('8', '⁸').replace('9', '⁹').replace('+', '⁺').replace('-', '⁻')

def _digits_to_normalscript(val):
    val = f'{val}'.replace('⁰', '0').replace('¹', '1').replace('²', '2').replace('³', '3')
    val = val.replace('⁴', '4').replace('⁵', '5').replace('⁶', '6').replace('⁷', '7')
    return val.replace('⁸', '8').replace('⁹', '9').replace('⁺', '+').replace('⁻', '-')

def _to_normal_script(val):
    return _digits_to_normalscript(val).replace('ₛ', 's').replace('ₑ', 'e')


# Dynamically set mass_table from xtrack.particles.masses to avoid circular imports and loops
_mass_table = {}
def _get_mass_table():
    if _mass_table == {}:
        for name, mass in mass__dict__.items():
            if name.endswith('_MASS_EV') and name != 'U_MASS_EV':
                # Mass definitions for charged particles are without the charge identifier,
                # so we need to try these first.
                try:
                    pdg_id = get_pdg_id_from_name(f'{name[:-8]}+')
                except ValueError:
                    pdg_id = get_pdg_id_from_name(name[:-8])
                if pdg_id not in _mass_table:
                    _mass_table[pdg_id] = mass
                else:
                    raise ValueError(f"Duplicate mass in Xtrack for particle {pdg_id}!")
    return _mass_table


def _mass_consistent(pdg_id, m, mask=None):
    if hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str) \
    and hasattr(m, '__len__') and not isinstance(m, str):
        if mask is None:
            return np.all([_mass_consistent(pdg, mm) for pdg, mm in zip(pdg_id, m)])
        else:
            pdg_id = np.array(pdg_id)[mask]
            m = np.array(m)[mask]
            return np.all([_mass_consistent(pdg, mm) for pdg, mm in zip(pdg_id, m)])
    elif hasattr(pdg_id, '__len__') and not isinstance(pdg_id, str):
        if mask is None:
            return np.all([_mass_consistent(pdg, m) for pdg in pdg_id])
        else:
            pdg_id = np.array(pdg_id)[mask]
            return np.all([_mass_consistent(pdg, m) for pdg in pdg_id])
    elif hasattr(m, '__len__') and not isinstance(m, str):
        if mask is None:
            return np.all([_mass_consistent(pdg_id, mm) for mm in m])
        else:
            m = np.array(m)[mask]
            return np.all([_mass_consistent(pdg_id, mm) for mm in m])

    try:
        q, _, _, _ = get_properties_from_pdg_id(pdg_id)
        return pdg_id == get_pdg_id_from_mass_charge(m, q)
    except ValueError:
        # No check if mass cannot be retrieved
        return True


# Make sure no duouble names exist in the pdg_table, after removing subscripts
# and going to lower case
def _check_pdg_table():
    names = [vvv for vv in pdg_table.values() for vvv in vv[1:]]
    names = [_to_normal_script(name).lower() for name in names]
    if len(names) != len(np.unique(names)):
        double_names = []
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                if _to_normal_script(name1).lower() \
                == _to_normal_script(name2).lower():
                    double_names.append(name1)
                    double_names.append(name2)
        raise ValueError(f"Duplicate names in pdg_table:\n{double_names}")

_check_pdg_table()
