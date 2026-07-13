"""
Q-Score: Quasi-Identifier Detection and Risk Assessment with NLP Enhancement
=============================================================================
Detects quasi-identifiers using spaCy NER + pattern matching and calculates
re-identification risk based on k-anonymity principles.

All population statistics are fetched live from:
  - Census Bureau API (age, gender, state/location, ZIP population)
  - BLS API via CDC NIOCCS (occupation employment)
  - CDC Open Data / Chronic Disease Indicators (common disease prevalence)
  - Orphanet / Orphadata API (rare disease prevalence)
"""
import os
import re
import json
import hashlib
import time
import logging
import requests
import spacy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# =============================================================================
# Data Classes and Enums
# =============================================================================
class QIType(Enum):
    AGE = "age"
    DATE_OF_BIRTH = "date_of_birth"
    GENDER = "gender"
    LOCATION = "location"
    ZIP_CODE = "zip_code"
    DISEASE = "disease"
    OCCUPATION = "occupation"
    ETHNICITY = "ethnicity"

@dataclass
class QuasiIdentifier:
    qi_type: QIType
    raw_value: str
    normalized_value: str
    confidence: float
    start_pos: int
    end_pos: int
    detection_method: str = "pattern"

    def __hash__(self):
        return hash((self.qi_type, self.normalized_value))

    def __eq__(self, other):
        if not isinstance(other, QuasiIdentifier):
            return NotImplemented
        return (self.qi_type, self.normalized_value) == (other.qi_type, other.normalized_value)

@dataclass
class QIFrequency:
    qi_type: QIType
    value: str
    probability: float
    population_count: Optional[int] = None
    source: str = "unknown"

@dataclass
class QScoreResult:
    q_score: float
    expected_k: float
    detected_qis: list
    frequencies: dict
    explanation: str
    original_text: str
    masked_text: str

# =============================================================================
# Disk-based TTL cache for API responses
# =============================================================================
class _DiskCache:
    """Simple JSON file cache with TTL, stored in a .q_score_cache directory."""

    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: int = 86400):
        self._dir = Path(cache_dir or os.path.join(os.path.dirname(__file__), ".q_score_cache"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds

    def _key_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self._dir / f"{h}.json"

    def get(self, key: str):
        p = self._key_path(key)
        if p.exists():
            try:
                blob = json.loads(p.read_text())
                if time.time() - blob.get("ts", 0) < self._ttl:
                    return blob["data"]
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def set(self, key: str, data):
        p = self._key_path(key)
        p.write_text(json.dumps({"ts": time.time(), "data": data}))

_cache = _DiskCache()

# =============================================================================
# FIPS lookup (needed by Census API)
# =============================================================================
STATE_NAME_TO_FIPS = {
    'alabama': '01', 'alaska': '02', 'arizona': '04', 'arkansas': '05',
    'california': '06', 'colorado': '08', 'connecticut': '09', 'delaware': '10',
    'district of columbia': '11', 'florida': '12', 'georgia': '13', 'hawaii': '15',
    'idaho': '16', 'illinois': '17', 'indiana': '18', 'iowa': '19', 'kansas': '20',
    'kentucky': '21', 'louisiana': '22', 'maine': '23', 'maryland': '24',
    'massachusetts': '25', 'michigan': '26', 'minnesota': '27', 'mississippi': '28',
    'missouri': '29', 'montana': '30', 'nebraska': '31', 'nevada': '32',
    'new hampshire': '33', 'new jersey': '34', 'new mexico': '35', 'new york': '36',
    'north carolina': '37', 'north dakota': '38', 'ohio': '39', 'oklahoma': '40',
    'oregon': '41', 'pennsylvania': '42', 'rhode island': '44',
    'south carolina': '45', 'south dakota': '46', 'tennessee': '47', 'texas': '48',
    'utah': '49', 'vermont': '50', 'virginia': '51', 'washington': '53',
    'west virginia': '54', 'wisconsin': '55', 'wyoming': '56',
}

STATE_ABBREV_TO_NAME = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'dc': 'district of columbia', 'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii',
    'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming',
}

US_STATES = set(STATE_NAME_TO_FIPS.keys())

# =============================================================================
# NLP-Enhanced Quasi-Identifier Extractor
# =============================================================================
class NLPQIExtractor:
    """
    Extracts quasi-identifiers using a dual-model approach:
      1. General spaCy (en_core_web_lg) for structural QIs (age, gender, location).
      2. SciSpaCy (en_ner_bc5cdr_md) for highly accurate disease detection.
      3. Context-based regex and PhraseMatcher for occupations.
    """

    def __init__(self, model_name: str = "en_core_web_lg", disease_model: str = "en_ner_bc5cdr_md"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        try:
            self.nlp_disease = spacy.load(disease_model)
        except OSError:
            log.warning(f"Could not load {disease_model}. Diseases may not be detected. Run 'pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz' to fix.")
            self.nlp_disease = None

        self._add_structural_patterns()
        self._setup_occupation_matcher()

    def _add_structural_patterns(self):
        """Add patterns only for structured/syntactic QIs (age, gender)."""
        ruler = self.nlp.add_pipe("entity_ruler", before="ner",
                                  config={"overwrite_ents": False})
        ruler.add_patterns([
            {"label": "AGE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["year", "years"]}}, {"LOWER": "old"}]},
            {"label": "AGE", "pattern": [{"LIKE_NUM": True}, {"TEXT": {"REGEX": r"y\.?o\.?"}}]},
            {"label": "AGE", "pattern": [{"LOWER": "age"}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True}]},
            {"label": "GENDER", "pattern": [{"LOWER": {"IN": ["male", "female", "man", "woman", "boy", "girl"]}}]},
        ])

    def _setup_occupation_matcher(self):
        from spacy.matcher import PhraseMatcher
        self.occ_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        # A small sample of common occupations that might appear without context cues
        common_occupations = [
            "doctor", "physician", "nurse", "cardiologist", "neurologist", "surgeon",
            "lawyer", "attorney", "teacher", "professor", "engineer", "developer",
            "programmer", "firefighter", "police officer", "pilot", "chef", "accountant",
            "dentist", "pharmacist", "therapist", "scientist", "researcher", "manager",
            "consultant", "analyst", "mechanic", "electrician", "plumber"
        ]
        patterns = [self.nlp.make_doc(text) for text in common_occupations]
        self.occ_matcher.add("OCCUPATION", patterns)

    # Regex fallbacks
    AGE_PATTERNS = [
        r'\b(\d{1,3})\s*[-–]?\s*(?:years?\s*old|year\s*old|yo|y\.o\.|y/o)\b',
        r'\bage[d]?\s*[:;]?\s*(\d{1,3})\b',
        r'\b(\d{1,3})\s*(?:year|yr)[\s-]*old\b',
        r'\b([1-9][0-9]?)\s*[MFmf]\b',
    ]
    DOB_PATTERNS = [
        r'\b(?:DOB|D\.O\.B\.|Date of Birth|Birth\s*Date|Born)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(?:DOB|D\.O\.B\.|Date of Birth|Birth\s*Date|Born)[:\s]*(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
    ]
    GENDER_MAP = {
        'male': 'male', 'm': 'male', 'man': 'male', 'boy': 'male',
        'female': 'female', 'f': 'female', 'woman': 'female', 'girl': 'female',
    }

    # Context-based occupation extraction patterns.
    OCCUPATION_CONTEXT_PATTERNS = [
        r'(?:works?\s+as\s+(?:a|an)?|employed\s+as\s+(?:a|an)?'
        r'|occupation[:\s]+|job[:\s]+|profession[:\s]+'
        r'|career[:\s]+|position[:\s]+|role[:\s]+)\s*'
        r'((?:[A-Za-z]+(?:\s+[A-Za-z]+){0,2}))',
        # "is a/an <occupation>" heuristic for titles ending in common suffixes
        r'\bis\s+(?:a|an)\s+((?:[A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?:ist|er|or|ian|ant|ent|man|eer|ive|ot|geon|cher|yst|ner))\b',
    ]

    _STOP_WORDS = {
        'and', 'or', 'but', 'the', 'a', 'an', 'in', 'on', 'at', 'to',
        'for', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'with', 'from', 'that',
        'this', 'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
        'if', 'then', 'than', 'so', 'very', 'not', 'no', 'he', 'she', 'it',
        'they', 'we', 'you', 'i', 'me', 'him', 'her', 'his', 'my', 'your',
        'our', 'its', 'also', 'of', 'by', 'as',
    }

    # ------------------------------------------------------------------
    def extract(self, text: str) -> List[QuasiIdentifier]:
        doc = self.nlp(text)
        qis = []
        qis.extend(self._from_general_ner(doc))
        qis.extend(self._from_disease_ner(text))
        qis.extend(self._from_occupation_matcher(doc))
        qis.extend(self._occupations_regex(text))
        qis.extend(self._ages_regex(text))
        qis.extend(self._dobs_regex(text))
        qis.extend(self._zip_regex(text))
        return self._dedup(qis)

    def _from_disease_ner(self, text: str) -> List[QuasiIdentifier]:
        if not self.nlp_disease:
            return []
        doc = self.nlp_disease(text)
        qis = []
        for ent in doc.ents:
            if ent.label_ == "DISEASE":
                qis.append(QuasiIdentifier(QIType.DISEASE, ent.text, ent.text.lower(), 
                                           0.9, ent.start_char, ent.end_char, "scispacy_ner"))
        return qis

    def _from_occupation_matcher(self, doc) -> List[QuasiIdentifier]:
        qis = []
        matches = self.occ_matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            qis.append(QuasiIdentifier(QIType.OCCUPATION, span.text, span.text.lower(),
                                       0.85, span.start_char, span.end_char, "phrase_matcher"))
        return qis

    def _trim_capture(self, raw: str) -> str:
        words = raw.strip().split()
        while words and words[-1].lower().rstrip('.,;:') in self._STOP_WORDS:
            words.pop()
        return ' '.join(words).strip().rstrip('.,;:')

    def _occupations_regex(self, text: str) -> List[QuasiIdentifier]:
        out: List[QuasiIdentifier] = []
        for pat in self.OCCUPATION_CONTEXT_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                name = self._trim_capture(m.group(1))
                if len(name) >= 3:
                    out.append(QuasiIdentifier(QIType.OCCUPATION, m.group(0).strip(), name.lower(),
                                               0.8, m.start(), m.end(), "pattern"))
        return out

    def _from_general_ner(self, doc) -> List[QuasiIdentifier]:
        qis: List[QuasiIdentifier] = []
        for ent in doc.ents:
            qi = None
            label = ent.label_
            if label == "AGE":
                m = re.search(r'\d+', ent.text)
                if m:
                    age = int(m.group())
                    if 0 <= age <= 120:
                        qi = QuasiIdentifier(QIType.AGE, ent.text, str(age),
                                             0.9, ent.start_char, ent.end_char, "ner")
            elif label == "GENDER":
                g = self.GENDER_MAP.get(ent.text.lower())
                if g:
                    qi = QuasiIdentifier(QIType.GENDER, ent.text, g,
                                         0.95, ent.start_char, ent.end_char, "ner")
            elif label in ("GPE", "LOC"):
                loc = ent.text.lower()
                normalized = STATE_ABBREV_TO_NAME.get(loc, loc)
                if normalized in US_STATES:
                    qi = QuasiIdentifier(QIType.LOCATION, ent.text, normalized, 0.85, ent.start_char, ent.end_char, "ner")
                else:
                    qi = QuasiIdentifier(QIType.LOCATION, ent.text, loc, 0.7, ent.start_char, ent.end_char, "ner")
            elif label == "NORP":
                qi = QuasiIdentifier(QIType.ETHNICITY, ent.text, ent.text.lower(), 0.7, ent.start_char, ent.end_char, "ner")
            
            if qi:
                qis.append(qi)
        return qis

    def _ages_regex(self, text: str) -> List[QuasiIdentifier]:
        out: List[QuasiIdentifier] = []
        for pat in self.AGE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                try:
                    age = int(m.group(1))
                    if 0 <= age <= 120:
                        out.append(QuasiIdentifier(QIType.AGE, m.group(0),
                                   str(age), 0.9, m.start(), m.end(), "pattern"))
                except ValueError:
                    continue
        return out

    def _dobs_regex(self, text: str) -> List[QuasiIdentifier]:
        out: List[QuasiIdentifier] = []
        for pat in self.DOB_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                out.append(QuasiIdentifier(QIType.DATE_OF_BIRTH, m.group(0),
                           m.group(1), 0.95, m.start(), m.end(), "pattern"))
        return out

    def _zip_regex(self, text: str) -> List[QuasiIdentifier]:
        out: List[QuasiIdentifier] = []
        # ZIP with context
        for m in re.finditer(
            r'(?:zip|postal|address|located|lives|resides)[^0-9]{0,30}(\d{5})(?:-\d{4})?',
            text, re.IGNORECASE):
            out.append(QuasiIdentifier(QIType.ZIP_CODE, m.group(1),
                       m.group(1), 0.95, m.start(1), m.end(1), "pattern"))
        # State abbrev + ZIP
        for m in re.finditer(r'\b[A-Z]{2}\s+(\d{5})(?:-\d{4})?\b', text):
            z = m.group(1)
            if not any(q.normalized_value == z for q in out):
                out.append(QuasiIdentifier(QIType.ZIP_CODE, m.group(0),
                           z, 0.9, m.start(1), m.end(1), "pattern"))
        return out

    @staticmethod
    def _dedup(qis: List[QuasiIdentifier]) -> List[QuasiIdentifier]:
        best: dict[tuple, tuple[QuasiIdentifier, int]] = {}
        for qi in qis:
            key = (qi.qi_type, qi.normalized_value)
            pri = 1 if qi.detection_method == "ner" else 0
            if key not in best or pri > best[key][1]:
                best[key] = (qi, pri)
        return [qi for qi, _ in best.values()]


# =============================================================================
# API-based Population Data Sources
# =============================================================================
class PopulationDataSource(ABC):
    @abstractmethod
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        pass


class CensusDataSource(PopulationDataSource):
    """
    Fetches population statistics live from the Census Bureau API.
      - Age:      ACS 5-year B01001 (Sex by Age) groups
      - Gender:   ACS 5-year B01001 male/female totals
      - Location: ACS 5-year B01003 (Total Population) per state
      - ZIP code: ACS 5-year B01003 per ZCTA
    Total US population is also fetched from ACS, not hardcoded.
    """
    BASE = "https://api.census.gov/data"
    YEAR = "2023"
    TIMEOUT = 15

    # B01001 variable codes — male groups (_003 to _025), female (_027 to _049)
    # Each tuple: (variable_suffix, age_low, age_high)
    _AGE_GROUPS_MALE = [
        ("003", 0, 4), ("004", 5, 9), ("005", 10, 14), ("006", 15, 17),
        ("007", 18, 19), ("008", 20, 20), ("009", 21, 21), ("010", 22, 24),
        ("011", 25, 29), ("012", 30, 34), ("013", 35, 39), ("014", 40, 44),
        ("015", 45, 49), ("016", 50, 54), ("017", 55, 59), ("018", 60, 61),
        ("019", 62, 64), ("020", 65, 66), ("021", 67, 69), ("022", 70, 74),
        ("023", 75, 79), ("024", 80, 84), ("025", 85, 120),
    ]
    _AGE_GROUPS_FEMALE = [
        ("027", 0, 4), ("028", 5, 9), ("029", 10, 14), ("030", 15, 17),
        ("031", 18, 19), ("032", 20, 20), ("033", 21, 21), ("034", 22, 24),
        ("035", 25, 29), ("036", 30, 34), ("037", 35, 39), ("038", 40, 44),
        ("039", 45, 49), ("040", 50, 54), ("041", 55, 59), ("042", 60, 61),
        ("043", 62, 64), ("044", 65, 66), ("045", 67, 69), ("046", 70, 74),
        ("047", 75, 79), ("048", 80, 84), ("049", 85, 120),
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CENSUS_API_KEY", "")

    # -- helpers --------------------------------------------------------
    def _census_get(self, dataset: str, variables: str, geo: str) -> Optional[list]:
        """Make a Census API GET request and return parsed JSON rows."""
        cache_key = f"census|{dataset}|{variables}|{geo}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        url = f"{self.BASE}/{self.YEAR}/{dataset}"
        params = {"get": variables, "for": geo}
        if self.api_key:
            params["key"] = self.api_key
        try:
            r = requests.get(url, params=params, timeout=self.TIMEOUT)
            r.raise_for_status()
            data = r.json()
            _cache.set(cache_key, data)
            return data
        except Exception as exc:
            log.warning("Census API error: %s", exc)
            return None

    @lru_cache(maxsize=1)
    def _us_total_population(self) -> int:
        """Fetch total US population from ACS B01003_001E."""
        data = self._census_get("acs/acs5", "B01003_001E", "us:1")
        if data and len(data) > 1:
            try:
                return int(data[1][0])
            except (ValueError, IndexError):
                pass
        return 331_900_000  # safety fallback only if API is unreachable

    # -- public ---------------------------------------------------------
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        if qi_type == QIType.AGE:
            return self._age(value)
        if qi_type == QIType.DATE_OF_BIRTH:
            return self._dob(value)
        if qi_type == QIType.GENDER:
            return self._gender(value)
        if qi_type == QIType.LOCATION:
            return self._location(value)
        if qi_type == QIType.ZIP_CODE:
            return self._zip(value)
        return None

    # -- age -----------------------------------------------------------
    def _age(self, age_str: str) -> Optional[QIFrequency]:
        try:
            age = int(age_str)
        except ValueError:
            return None
        if age < 0 or age > 120:
            return None

        # Build variable list from B01001 (both sexes combined)
        all_groups = self._AGE_GROUPS_MALE + self._AGE_GROUPS_FEMALE
        # Find the group(s) this age falls into
        matching_vars = [f"B01001_{s}E" for s, lo, hi in all_groups if lo <= age <= hi]
        if not matching_vars:
            return None

        # Also get total population variable
        var_list = ",".join(matching_vars + ["B01001_001E"])
        data = self._census_get("acs/acs5", var_list, "us:1")
        if not data or len(data) < 2:
            return None

        try:
            header = data[0]
            row = data[1]
            total_idx = header.index("B01001_001E")
            total = int(row[total_idx])
            group_count = sum(int(row[header.index(v)]) for v in matching_vars)
            # group_count covers a range; estimate single-year fraction
            span = 0
            for s, lo, hi in all_groups:
                if lo <= age <= hi:
                    span += (hi - lo + 1)
            # Combined male+female groups matched => divide by 2 span widths
            single_year_count = group_count / max(span // 2, 1) if span else group_count
            prob = single_year_count / total if total else 0.01
            return QIFrequency(QIType.AGE, age_str, prob,
                               int(single_year_count), "Census Bureau ACS B01001")
        except Exception:
            return None

    # -- date of birth (treat as age) ----------------------------------
    def _dob(self, dob_str: str) -> Optional[QIFrequency]:
        # Convert DOB to age and delegate
        from datetime import datetime
        for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%Y/%m/%d",
                    "%m/%d/%y", "%m-%d-%y"):
            try:
                born = datetime.strptime(dob_str, fmt)
                today = datetime.now()
                age = today.year - born.year - (
                    (today.month, today.day) < (born.month, born.day))
                freq = self._age(str(age))
                if freq:
                    # DOB is much more specific than age alone — adjust
                    freq.probability /= 365.25
                    freq.population_count = int(
                        freq.population_count / 365.25) if freq.population_count else 0
                    freq.source = "Census Bureau ACS B01001 (DOB adjusted)"
                return freq
            except ValueError:
                continue
        return None

    # -- gender --------------------------------------------------------
    def _gender(self, gender: str) -> Optional[QIFrequency]:
        # B01001_002E = total male, B01001_026E = total female
        var_map = {"male": "B01001_002E", "female": "B01001_026E"}
        var = var_map.get(gender.lower())
        if not var:
            return None
        data = self._census_get("acs/acs5", f"{var},B01001_001E", "us:1")
        if not data or len(data) < 2:
            return None
        try:
            header, row = data[0], data[1]
            count = int(row[header.index(var)])
            total = int(row[header.index("B01001_001E")])
            prob = count / total if total else 0.5
            return QIFrequency(QIType.GENDER, gender, prob, count,
                               "Census Bureau ACS B01001")
        except Exception:
            return None

    # -- location (US state) -------------------------------------------
    def _location(self, location: str) -> Optional[QIFrequency]:
        loc = location.lower().strip()
        loc = STATE_ABBREV_TO_NAME.get(loc, loc)
        fips = STATE_NAME_TO_FIPS.get(loc)
        if not fips:
            return None  # non-state locations not supported via Census
        data = self._census_get("acs/acs5", "B01003_001E", f"state:{fips}")
        total_pop = self._us_total_population()
        if not data or len(data) < 2:
            return None
        try:
            state_pop = int(data[1][0])
            prob = state_pop / total_pop if total_pop else 0.02
            return QIFrequency(QIType.LOCATION, location, prob, state_pop,
                               "Census Bureau ACS B01003")
        except Exception:
            return None

    # -- ZIP / ZCTA ----------------------------------------------------
    def _zip(self, zip_code: str) -> Optional[QIFrequency]:
        data = self._census_get(
            "acs/acs5", "B01003_001E",
            f"zip code tabulation area:{zip_code}")
        total_pop = self._us_total_population()
        if not data or len(data) < 2:
            return None
        try:
            zcta_pop = int(data[1][0])
            prob = zcta_pop / total_pop if total_pop else 7500 / 331_900_000
            return QIFrequency(QIType.ZIP_CODE, zip_code, prob, zcta_pop,
                               "Census Bureau ACS ZCTA B01003")
        except Exception:
            return None


class OccupationDataSource(PopulationDataSource):
    """
    Two-step live lookup:
      1. CDC NIOCCS Autocoder: occupation text → SOC code
      2. BLS OES API v2: SOC code → national employment count
    """
    NIOCCS_URL = "https://wwwn.cdc.gov/nioccs/IOCode"
    BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    TIMEOUT = 15

    def __init__(self, bls_key: Optional[str] = None):
        self.bls_key = bls_key or os.getenv("BLS_API_KEY", "")

    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        if qi_type != QIType.OCCUPATION:
            return None
        soc = self._name_to_soc(value)
        if not soc:
            return None
        employment = self._soc_employment(soc)
        if employment is None:
            return None
        # BLS total employed ~160M; fetch dynamically from all-occ series
        total = self._total_employment()
        prob = employment / total if total else employment / 160_000_000
        return QIFrequency(QIType.OCCUPATION, value, prob, employment,
                           f"BLS OES via NIOCCS (SOC {soc})")

    # -- Step 1: occupation text → SOC via CDC NIOCCS -------------------
    def _name_to_soc(self, occupation: str) -> Optional[str]:
        cache_key = f"nioccs|{occupation.lower().strip()}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached if cached != "__NONE__" else None
        try:
            r = requests.get(self.NIOCCS_URL,
                             params={"o": occupation, "n": "1"},
                             timeout=self.TIMEOUT)
            r.raise_for_status()
            data = r.json()
            occ_list = data.get("Occupation", [])
            if occ_list:
                code = occ_list[0].get("Code", "")
                _cache.set(cache_key, code)
                return code
        except Exception as exc:
            log.warning("NIOCCS API error for '%s': %s", occupation, exc)
        _cache.set(cache_key, "__NONE__")
        return None

    # -- Step 2: SOC code → employment via BLS OES ----------------------
    def _soc_employment(self, soc_code: str) -> Optional[int]:
        """Query BLS OES for total employment of a given SOC code nationally."""
        # Build OES series ID:
        # OE U N 0000000 000000 XXXXXX 01
        # where XXXXXX is the 6-digit SOC (strip hyphens)
        soc_digits = soc_code.replace("-", "").replace(".", "")
        if len(soc_digits) < 6:
            soc_digits = soc_digits.ljust(6, "0")
        series_id = f"OEUN0000000000000{soc_digits[:6]}01"

        cache_key = f"bls|{series_id}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached if cached != "__NONE__" else None

        try:
            payload = {
                "seriesid": [series_id],
                "startyear": str(int(CensusDataSource.YEAR) - 1),
                "endyear": CensusDataSource.YEAR,
            }
            if self.bls_key:
                payload["registrationkey"] = self.bls_key
            r = requests.post(self.BLS_URL, json=payload,
                              headers={"Content-type": "application/json"},
                              timeout=self.TIMEOUT)
            r.raise_for_status()
            js = r.json()
            if js.get("status") == "REQUEST_SUCCEEDED":
                for series in js["Results"]["series"]:
                    for item in series["data"]:
                        val = int(item["value"].replace(",", ""))
                        _cache.set(cache_key, val)
                        return val
        except Exception as exc:
            log.warning("BLS API error for SOC %s: %s", soc_code, exc)

        _cache.set(cache_key, "__NONE__")
        return None

    @lru_cache(maxsize=1)
    def _total_employment(self) -> int:
        """Total employment across all occupations (SOC 00-0000)."""
        val = self._soc_employment("00-0000")
        return val if val else 160_000_000


class DiseaseDataSource(PopulationDataSource):
    """
    Fetches disease prevalence from two live sources:
      1. CDC Chronic Disease Indicators (data.cdc.gov) for common diseases
      2. Orphanet / Orphadata API for rare diseases
    """
    # CDC CDI Socrata resource ID
    CDC_CDI_RESOURCE = "hksd-2xuw"
    CDC_BASE = "https://data.cdc.gov/resource"
    # Orphadata cross-referencing endpoint
    ORPHADATA_BASE = "https://api.orphadata.com"
    TIMEOUT = 15

    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        if qi_type != QIType.DISEASE:
            return None
        disease = value.lower().strip()

        # Try CDC first (common chronic diseases)
        freq = self._cdc_prevalence(disease)
        if freq:
            return freq

        # Fall back to Orphanet (rare diseases)
        freq = self._orphanet_prevalence(disease)
        if freq:
            return freq

        return None

    # -- CDC Chronic Disease Indicators --------------------------------
    def _cdc_prevalence(self, disease: str) -> Optional[QIFrequency]:
        """Search CDC CDI dataset for a prevalence-type indicator matching
        the disease name. Returns overall US crude prevalence if found."""
        cache_key = f"cdc_cdi|{disease}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return QIFrequency(**cached) if cached != "__NONE__" else None

        url = f"{self.CDC_BASE}/{self.CDC_CDI_RESOURCE}.json"
        
        # Escape single quotes in disease name for SOQL
        safe_disease = disease.replace("'", "''")
        
        # SoQL: search topic by disease name, get Overall prevalence
        # Use $where with UPPER/LOWER for case-insensitive partial match
        params = {
            "$where": (
                f"lower(topic) like '%{safe_disease}%' "
                f"AND lower(datavaluetype) like '%crude prevalence%' "
                f"AND locationabbr = 'US'"
            ),
            "$order": "yearend DESC",
            "$limit": 5,
        }
        try:
            r = requests.get(url, params=params, timeout=self.TIMEOUT)
            r.raise_for_status()
            rows = r.json()
            for row in rows:
                dv = row.get("datavalue")
                if dv:
                    prevalence_pct = float(dv)
                    prob = prevalence_pct / 100.0
                    pop_count = int(prob * 331_900_000)
                    freq = QIFrequency(QIType.DISEASE, disease, prob,
                                       pop_count,
                                       f"CDC Chronic Disease Indicators ({row.get('yearend', '?')})")
                    _cache.set(cache_key, {
                        "qi_type": freq.qi_type.value,
                        "value": freq.value,
                        "probability": freq.probability,
                        "population_count": freq.population_count,
                        "source": freq.source,
                    })
                    return freq
        except Exception as exc:
            log.warning("CDC CDI API error for '%s': %s", disease, exc)

        _cache.set(cache_key, "__NONE__")
        return None

    # -- Orphanet / Orphadata (rare diseases) --------------------------
    def _orphanet_prevalence(self, disease: str) -> Optional[QIFrequency]:
        """
        1. Search Orphadata cross-referencing to find ORPHAcode by name
        2. Use the epidemiology data for that code
        """
        cache_key = f"orphanet|{disease}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return QIFrequency(**cached) if cached != "__NONE__" else None

        orphacode = self._find_orphacode(disease)
        if not orphacode:
            _cache.set(cache_key, "__NONE__")
            return None

        # Query epidemiology for this ORPHAcode
        try:
            url = (f"{self.ORPHADATA_BASE}/rd-epidemiology"
                   f"/orphacodes/{orphacode}")
            r = requests.get(url, timeout=self.TIMEOUT,
                             headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()
            # Parse prevalence from response
            prob = self._parse_orphanet_prevalence(data)
            if prob is not None:
                pop_count = int(prob * 331_900_000)
                freq = QIFrequency(QIType.DISEASE, disease, prob,
                                   pop_count, f"Orphanet (ORPHA:{orphacode})")
                _cache.set(cache_key, {
                    "qi_type": freq.qi_type.value,
                    "value": freq.value,
                    "probability": freq.probability,
                    "population_count": freq.population_count,
                    "source": freq.source,
                })
                return freq
        except Exception as exc:
            log.warning("Orphadata epidemiology error for '%s': %s",
                        disease, exc)

        _cache.set(cache_key, "__NONE__")
        return None

    def _find_orphacode(self, disease: str) -> Optional[str]:
        """Use Orphadata cross-referencing to find ORPHAcode from name."""
        try:
            url = (f"{self.ORPHADATA_BASE}/rd-cross-referencing"
                   f"/orphacodes/names/{requests.utils.quote(disease)}")
            r = requests.get(url, timeout=self.TIMEOUT,
                             headers={"Accept": "application/json"})
            if r.status_code == 200:
                data = r.json()
                # Handle list or dict response
                results = data if isinstance(data, list) else data.get("results", [data])
                if results:
                    entry = results[0] if isinstance(results, list) else results
                    return str(entry.get("ORPHAcode", entry.get("orphacode", "")))
        except Exception as exc:
            log.warning("Orphadata name lookup error for '%s': %s", disease, exc)
        return None

    @staticmethod
    def _parse_orphanet_prevalence(data: dict) -> Optional[float]:
        """
        Parse prevalence from Orphadata epidemiology response.
        Orphanet uses prevalence classes like "1-9 / 100 000".
        """
        PREVALENCE_MAP = {
            ">1 / 1000": 1 / 500,
            "1-5 / 10 000": 3 / 10_000,
            "1-9 / 10 000": 5 / 10_000,
            "6-9 / 10 000": 7.5 / 10_000,
            "1-9 / 100 000": 5 / 100_000,
            "1-9 / 1 000 000": 5 / 1_000_000,
            "<1 / 1 000 000": 0.5 / 1_000_000,
        }
        try:
            # Navigate various response shapes
            prevalences = []
            if isinstance(data, dict):
                epi = data.get("Epidemiology", data)
                if isinstance(epi, dict):
                    prevalences = epi.get("Prevalences", epi.get("prevalences", []))
                elif isinstance(epi, list):
                    prevalences = epi

            for prev in prevalences:
                pclass = prev.get("PrevalenceClass",
                                  prev.get("prevalenceClass", ""))
                ptype = prev.get("PrevalenceType",
                                 prev.get("prevalenceType", ""))
                # Prefer "Point prevalence" over "Birth prevalence"
                if "point" in str(ptype).lower() or "prevalence" in str(ptype).lower():
                    for key, val in PREVALENCE_MAP.items():
                        if key.lower() in pclass.lower():
                            return val
            # If no match, try first prevalence class anyway
            if prevalences:
                pclass = prevalences[0].get("PrevalenceClass",
                    prevalences[0].get("prevalenceClass", ""))
                for key, val in PREVALENCE_MAP.items():
                    if key.lower() in pclass.lower():
                        return val
        except Exception:
            pass
        return None


# =============================================================================
# Q-Score Calculator with Masking
# =============================================================================
class QScoreCalculator:
    """Calculates Q-Score based on k-anonymity principles."""

    def __init__(self, k_threshold: int = 5):
        self.census = CensusDataSource()
        self.disease = DiseaseDataSource()
        self.occupation = OccupationDataSource()
        self.extractor = NLPQIExtractor()
        self.k_threshold = k_threshold

    @property
    def reference_population(self) -> int:
        return self.census._us_total_population()

    def calculate(self, text: str) -> QScoreResult:
        detected_qis = self.extractor.extract(text)
        ref_pop = self.reference_population

        if not detected_qis:
            return QScoreResult(0.0, ref_pop, [], {},
                                "No quasi-identifiers detected.", text, text)

        frequencies: dict[QuasiIdentifier, QIFrequency] = {}
        for qi in detected_qis:
            freq = self._get_frequency(qi)
            if freq:
                frequencies[qi] = freq

        if not frequencies:
            return QScoreResult(0.0, ref_pop, detected_qis, {},
                f"Detected {len(detected_qis)} QI(s) but could not lookup frequencies.",
                text, text)

        joint_prob = 1.0
        for f in frequencies.values():
            joint_prob *= f.probability

        expected_k = ref_pop * joint_prob
        if expected_k >= self.k_threshold:
            q_score = 0.0
            masked_text = text
        else:
            q_score = min(1.0, 1.0 - (expected_k / self.k_threshold))
            masked_text = self._mask(text, detected_qis)

        return QScoreResult(
            q_score=q_score,
            expected_k=expected_k,
            detected_qis=detected_qis,
            frequencies=frequencies,
            explanation=self._explain(detected_qis, frequencies, expected_k),
            original_text=text,
            masked_text=masked_text,
        )

    def _get_frequency(self, qi: QuasiIdentifier) -> Optional[QIFrequency]:
        if qi.qi_type in (QIType.AGE, QIType.DATE_OF_BIRTH, QIType.GENDER,
                          QIType.LOCATION, QIType.ZIP_CODE):
            return self.census.get_frequency(qi.qi_type, qi.normalized_value)
        if qi.qi_type == QIType.DISEASE:
            return self.disease.get_frequency(qi.qi_type, qi.normalized_value)
        if qi.qi_type == QIType.OCCUPATION:
            return self.occupation.get_frequency(qi.qi_type, qi.normalized_value)
        return None

    @staticmethod
    def _mask(text: str, qis: List[QuasiIdentifier]) -> str:
        for qi in sorted(qis, key=lambda q: q.start_pos, reverse=True):
            text = text[:qi.start_pos] + f"[{qi.qi_type.value.upper()}]" + text[qi.end_pos:]
        return text

    def _explain(self, qis, freqs, expected_k) -> str:
        lines = [f"Detected {len(qis)} quasi-identifier(s):"]
        for qi in qis:
            f = freqs.get(qi)
            tag = f"[{qi.detection_method}]"
            if f:
                lines.append(
                    f"  - {qi.qi_type.value}: '{qi.normalized_value}' {tag} "
                    f"(P ≈ {f.probability:.2e}, ~{f.population_count:,} people) "
                    f"[{f.source}]")
            else:
                lines.append(
                    f"  - {qi.qi_type.value}: '{qi.normalized_value}' {tag} "
                    f"(frequency unknown)")
        lines.append(f"\nExpected equivalence class size E[k]: {expected_k:.4f}")
        if expected_k < 1:
            lines.append("🚨 E[k] < 1: Likely UNIQUELY IDENTIFYING")
        elif expected_k < self.k_threshold:
            lines.append(f"⚠️ E[k] < {self.k_threshold}: HIGH re-identification risk")
        else:
            lines.append(f"✓ E[k] ≥ {self.k_threshold}: Acceptable k-anonymity")
        return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================
class QScoreAnalyzer:
    def __init__(self, k_threshold: int = 5):
        self.calculator = QScoreCalculator(k_threshold=k_threshold)

    def analyze(self, text: str) -> QScoreResult:
        return self.calculator.calculate(text)

    def get_score(self, text: str) -> float:
        return self.calculator.calculate(text).q_score

    def get_report(self, text: str) -> dict:
        result = self.calculator.calculate(text)
        return {
            "q_score": round(result.q_score, 4),
            "expected_k": round(result.expected_k, 4),
            "risk_level": ("HIGH" if result.q_score >= 0.7 else
                           "MEDIUM" if result.q_score >= 0.3 else "LOW"),
            "k_threshold": self.calculator.k_threshold,
            "detected_quasi_identifiers": [
                {
                    "type": qi.qi_type.value,
                    "raw_value": qi.raw_value,
                    "normalized_value": qi.normalized_value,
                    "confidence": qi.confidence,
                    "detection_method": qi.detection_method,
                    "position": {"start": qi.start_pos, "end": qi.end_pos},
                    "frequency": {
                        "probability": result.frequencies[qi].probability,
                        "population_count": result.frequencies[qi].population_count,
                        "source": result.frequencies[qi].source,
                    } if qi in result.frequencies else None,
                }
                for qi in result.detected_qis
            ],
            "explanation": result.explanation,
            "masked_text": result.masked_text,
        }


def calculate_qscore(text: str) -> dict:
    """Calculate Q-Score for text with masking."""
    return QScoreAnalyzer().get_report(text)


if __name__ == "__main__":
    sample = """
    Patient is a 45 year old male living in Maryland, zip 21218. He is a zoologist.
    He has two children. He has hypermobility.
    """
    report = calculate_qscore(sample)
    print(json.dumps(report, indent=2))