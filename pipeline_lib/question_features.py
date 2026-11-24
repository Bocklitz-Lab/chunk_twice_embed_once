"""Two AUTOMATIC, rule-based features of the question text: ``auto_has_number`` and ``auto_has_formula``.

They are NOT annotations and NOT question types. The dataset has no label for "the question contains a number or a chemical formula"
(Reviewer 2 asks about texts with formulas and numerical evidence), so they are computed here from the question text alone: deterministic,
no network, no model. Everywhere they carry the prefix ``auto_`` and ``feature_source = rule_based``; they never go into a column called ``type`` or
``category`` and are never merged with ``question_type`` (the official label of ``question_metadata.jsonl``). The rules are checked by hand on a
seeded sample of 50 questions (``tools/question_features_check.py``).

auto_has_number  the question contains a number: a run of digits with an optional decimal part (``3``, ``0.5``, ``2,5``), also in a range (``5-10``) or
                 glued to a unit (``25nm``, ``90%``, ``300 K``). A digit does NOT count as a number when it belongs to a name or a formula: it is
                 preceded by a letter, digit, underscore, ``^``, a bracket or a letter and a hyphen (``CO2``, ``H2O``, ``C60``, ``SO4^2-``, ``Ti3C2``, ``(OH)2``,
                 ``SARS-CoV-2``, ``mpGNS-3``), or it is followed directly by a letter that is not a unit (``3D``, ``1H NMR``, ``13C``), by a prime
                 (``1'-cyano``) or by a hyphen and a letter (locants: ``1-butanol``, ``2-methyl``). Years, integers and numbers that label a compound
                 (``compound 1``) count as numbers: the rule is lexical. Numbers written as words ("two") do not count.
auto_has_formula the question contains a token that parses COMPLETELY into element symbols with optional counts, groups and charges
                 (``H2O``, ``NaCl``, ``CO2``, ``Ca(OH)2``, ``(NH4)2SO4``, ``Fe3+``, ``SO4^2-``, ``Fe(II)``) and is a formula: at least two element symbols,
                 or one symbol with a count, a charge or an oxidation state in brackets. A single bare symbol (``In``, ``As``, ``No``, ``He``, ``I``, ``C``,
                 ``Fe``) is a word or an element name, not a formula. Abbreviations that happen to parse into symbols are excluded by the explicit list
                 ``NOT_FORMULAS`` (UV, IR, NIR, PVC, ...). A token is a run of letters, digits, brackets, ``^``, ``+`` and ``-`` between spaces or other
                 punctuation; if the whole run does not parse it is split at hyphens (``NaCl-based`` gives ``NaCl``), unless a part is only digits
                 (``SARS-CoV-2`` is a virus, not a formula). Unicode subscripts and superscripts are read as digits and charges (``TiO₂``, ``CO₃²⁻``).
                 Tokens of capital letters only, without count, charge or brackets, are ambiguous with acronyms (``CO`` and ``OH`` against ``SCO``
                 and ``VOC``): they count only if listed in ``CAPITALS_ONLY_FORMULAS``; an acronym defined in round brackets, ``(SF)``, never counts.
                 Known gaps and errors: formulas with a slash (``Pd/C``), with a letter that is no element (``Ti3C2Tx``) or written in words; a
                 symbol with a count that is a label (``S3 state``, ``C14 Laves``) is counted as a formula. The hand check measures the errors.

``question_features`` returns the two flags and the tokens that triggered them, so a person can check each flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

ELEMENTS = frozenset("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
                     "In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu "
                     "Am Cm Bk Cf Es Fm Md No Lr".split())

# abbreviations, roman numerals and words in capitals that parse into element symbols but are not formulas in a question
NOT_FORMULAS = frozenset({
    "UV", "IR", "NIR", "CV", "PVC", "PS", "PP", "PU", "ICP", "PCB", "BCS", "COF", "CNF", "CNC", "NP", "SI", "IC", "PI", "PH", "PB", "US", "UK", "OK", "IN",
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
    "CoV", "SSAs", "VOCs", "C++",                                 # SARS-CoV, sulfonated sulfoxide-like acronyms, volatile organic compounds, a language (seen in the questions)
})

# ambiguous tokens of capital letters only that are formulas (small molecules, ions and radicals); every other such token is taken as an acronym
CAPITALS_ONLY_FORMULAS = frozenset({"CO", "NO", "OH", "OOH", "HOO", "HCN", "CN", "NH", "CH", "HF", "HCl", "NOx", "SO", "HO", "HS", "NS", "PO", "BN", "SiC",
                                    "CS", "KI", "KBr", "HI", "BH", "CC", "OO", "NN", "CF"})
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SUPER_RUN = re.compile("[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")
_OXIDATION = r"\((?:I{1,3}|IV|VI{0,3}|IX|X)\)"
_CHARGE_END = re.compile(r"(\^\(?\d*[+\-−]\)?|\^[+\-−]\d+|\d*\++|\d*−)$")
_RUN = re.compile(r"[A-Za-z0-9()\[\]^+−\-]+")

UNITS = ("nm", "µm", "μm", "mm", "cm", "km", "mg", "µg", "μg", "kg", "mol", "mmol", "µmol", "μmol", "kJ", "kcal", "eV", "keV", "meV", "mV", "kV",
         "kHz", "MHz", "GHz", "THz", "Hz", "MPa", "GPa", "kPa", "Pa", "bar", "atm", "ppm", "ppb", "mL", "µL", "μL", "mM", "µM", "μM", "nM", "wt", "vol",
         "rpm", "min", "ms", "ns", "ps", "fs", "mA", "kW", "mW", "Å", "nmol", "cd", "lm")
_UNIT_ALT = "|".join(sorted((re.escape(u) for u in UNITS), key=len, reverse=True))
# one number: digits, optional decimal part, optional second number of a range
_NUM = r"\d+(?:[.,]\d+)?"
_NOT_AFTER = "A-Za-z0-9_^.,\u2212(){}\\[\\]"          # a digit after one of these belongs to a name, a formula or a longer number
NUMBER_RE = re.compile(
    rf"(?<![{_NOT_AFTER}])(?<!\d[.,])(?<![A-Za-z]-)(?<!\{{-)(?P<num>{_NUM}(?:\s?[\u2013\u2212-]\s?{_NUM})?)"
    rf"(?:\s?(?P<unit>{_UNIT_ALT}|°C|°F|°|%|K|M|g|h|s|L|m|J|W|V|A)(?![A-Za-z]))?"
    rf"(?![A-Za-z0-9_'\u2019\u2032])(?![.,]\d)(?!-[A-Za-z])")
_NUM_GLUED_UNIT = re.compile(rf"(?<![{_NOT_AFTER}])(?<![A-Za-z]-)(?P<num>{_NUM})(?P<unit>{_UNIT_ALT}|°C|°F|%)(?![A-Za-z])")


@dataclass(frozen=True)
class QuestionFeatures:
    auto_has_number: bool
    auto_has_formula: bool
    number_tokens: Tuple[str, ...]
    formula_tokens: Tuple[str, ...]


def number_tokens(question: str) -> List[str]:
    """The numbers (with their unit, if any) of ``question`` that make ``auto_has_number`` true, in text order."""
    out = []
    for m in NUMBER_RE.finditer(normalise(question)):
        out.append(m.group(0).strip())
    seen = {t for t in out}
    for m in _NUM_GLUED_UNIT.finditer(normalise(question)):          # "25nm", "90%": the general pattern refuses a digit followed by letters
        if m.group(0) not in seen:
            out.append(m.group(0))
    return out


# ------------------------------------------------------------------------------------------------------------------
# formulas
# ------------------------------------------------------------------------------------------------------------------
def _parse_group(s: str, i: int, closing: Optional[str]) -> Optional[Tuple[int, int, bool, bool]]:
    """Parse element symbols, counts and bracketed groups from ``s[i:]`` until ``closing`` (or the end). Returns (next index, number of symbols,
    any count, any oxidation state) or None if the text is not a formula."""
    n_sym, has_count, has_ox = 0, False, False
    start = i
    while i < len(s):
        c = s[i]
        if closing and c == closing:
            break
        if c in "([":
            ox = re.match(_OXIDATION, s[i:])
            if ox and n_sym and i > start and s[i - 1].isalpha():           # Fe(II): oxidation state right after a symbol
                has_ox = True
                i += ox.end()
                continue
            close = ")" if c == "(" else "]"
            sub = _parse_group(s, i + 1, close)
            if sub is None or sub[0] >= len(s) or s[sub[0]] != close or sub[1] == 0:
                return None
            i, n_sym = sub[0] + 1, n_sym + sub[1]
            has_count, has_ox = has_count or sub[2], has_ox or sub[3]
            m = re.match(r"\d+", s[i:])
            if m:
                has_count, i = True, i + m.end()
            continue
        if not c.isupper():
            return None
        two = s[i:i + 2]
        if len(two) == 2 and two[1].islower() and two in ELEMENTS:
            i += 2
        elif c in ELEMENTS:
            i += 1
        else:
            return None
        n_sym += 1
        m = re.match(r"\d+", s[i:])
        if m:
            has_count, i = True, i + m.end()
    return i, n_sym, has_count, has_ox


def normalise(question: str) -> str:
    """Unicode subscripts become digits (``CO₂`` -> ``CO2``) and a superscript run becomes a caret charge (``SO₄²⁻`` -> ``SO4^2-``)."""
    return _SUPER_RUN.sub(lambda m: "^" + m.group(0).translate(_SUP), question.translate(_SUB))


def is_formula(token: str) -> bool:
    """True if ``token`` is a chemical formula by the rule of the module docstring."""
    if token in NOT_FORMULAS or not token:
        return False
    base, has_charge = token, False
    m = _CHARGE_END.search(token)
    if m and m.start() > 0:
        base, has_charge = token[:m.start()], True
    if not base or not base[0].isupper() and base[0] not in "([":
        return False
    if base in NOT_FORMULAS:
        return False
    r = _parse_group(base, 0, None)
    if r is None or r[0] != len(base) or r[1] == 0:
        return False
    _, n_sym, has_count, has_ox = r
    if base.isalpha() and not has_charge and base.isupper() and base not in CAPITALS_ONLY_FORMULAS:
        return False                                                  # CAPITALS ONLY: an acronym unless listed
    return n_sym >= 2 or has_count or has_ox or has_charge


def formula_tokens(question: str) -> List[str]:
    """The tokens of ``question`` that make ``auto_has_formula`` true, in text order."""
    out = []
    for run in _RUN.findall(normalise(question)):
        run = run.lstrip("-\u2212") if _CHARGE_END.search(run) else run.strip("-\u2212+")     # a final "-" or "+" after "^2" / a count is a charge
        if re.fullmatch(r"\([A-Za-z]+s?\)", run):
            continue                                                  # "(SF)": an acronym that the question defines
        parts = [p for p in re.split(r"[-\u2212]", run) if p]
        cands = [run] if is_formula(run) else ([] if any(p.isdigit() for p in parts) else parts)
        for c in cands:
            c = c.strip("([") if c.count("(") + c.count("[") > c.count(")") + c.count("]") else c
            c = c.rstrip(")]") if c.count(")") + c.count("]") > c.count("(") + c.count("[") else c
            if is_formula(c):
                out.append(c)
    return out


def question_features(question: str) -> QuestionFeatures:
    nums, forms = number_tokens(question), formula_tokens(question)
    return QuestionFeatures(bool(nums), bool(forms), tuple(nums), tuple(forms))
