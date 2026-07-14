"""
Q-Score: Quasi-Identifier Detection and Risk Assessment with Local SLM
======================================================================
Detects quasi-identifiers using a local Small Language Model (SLM) via Ollama
and calculates re-identification risk based on k-anonymity principles.

The SLM replaces the previous spaCy NER + regex approach for better coverage
of non-standard QIs and more accurate normalization to match external dataset
nomenclature (Orphanet, CDC, Census, BLS).

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
# SLM-Based Quasi-Identifier Extractor  (Ollama)
# =============================================================================
class SLMQIExtractor:
    """
    Extracts quasi-identifiers using a local Small Language Model via Ollama.

    Advantages over the previous NLP/regex approach:
      1. Catches QIs that don't match rigid NER or regex patterns.
      2. Normalizes detected values to standard nomenclature so downstream
         API lookups (Orphanet, CDC, Census, BLS) match correctly.
      3. Single model handles all QI types — no separate spaCy + SciSpaCy
         + PhraseMatcher + regex stacks.

    Results are cached to disk so repeated calls on the same text are free.
    """

    VALID_QI_TYPES = {t.value for t in QIType}

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct-q4_K_M",
        ollama_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: int = 60,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature or 0.0
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, text: str) -> List[QuasiIdentifier]:
        """Extract quasi-identifiers from *text* using the SLM."""
        cache_key = f"slm_qi|{hashlib.sha256(text.encode()).hexdigest()[:32]}"
        cached = _cache.get(cache_key)
        if cached is not None:
            return [self._dict_to_qi(d) for d in cached]

        try:
            raw_response = self._call_ollama(self._build_prompt(text))
            parsed = self._parse_response(raw_response)
            qis = self._to_quasi_identifiers(parsed, text)
            _cache.set(cache_key, [self._qi_to_dict(qi) for qi in qis])
            return qis
        except Exception as exc:
            log.warning("SLM QI extraction failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Prompt engineering
    # ------------------------------------------------------------------
    def _build_prompt(self, text: str) -> str:  # noqa: D401
        return (
            "You are a quasi-identifier detection system for privacy risk "
            "analysis.  Analyze the following text and extract ALL quasi-"
            "identifiers — attributes that are not directly identifying on "
            "their own but could help re-identify a person when combined "
            "with other attributes.\n\n"
            "Quasi-identifier types to detect:\n"
            '- age: A person\'s age. Normalize to just the integer (e.g. "45").\n'
            "- date_of_birth: A date of birth. Normalize to MM/DD/YYYY.\n"
            '- gender: Gender or sex. Normalize to "male" or "female".\n'
            "- location: Geographic location, especially US states or cities. "
            "For US states normalize to the full lowercase state name "
            '(e.g. "maryland", "new york").\n'
            "- zip_code: ZIP or postal code. Normalize to the 5-digit ZIP "
            'string (e.g. "21218").\n'
            "- disease: Any medical condition, disease, disorder, or syndrome. "
            "Normalize to the standard medical name as it would appear "
            "in the Orphanet rare-disease database or the CDC chronic-disease "
            'database (e.g. "ehlers-danlos syndrome", '
            '"type 2 diabetes mellitus", "joint hypermobility syndrome"). '
            "Use the most specific standard name.\n"
            "- occupation: Job title or profession. Normalize to the standard "
            'occupation title (e.g. "zoologist", "registered nurse").\n'
            "- ethnicity: Ethnic, racial, or national group. Normalize to "
            'lowercase (e.g. "hispanic", "african american").\n\n'
            "TEXT TO ANALYZE:\n"
            f'"""\n{text}\n"""\n\n'
            "Return ONLY valid JSON in this exact format:\n"
            "{\n"
            '  "quasi_identifiers": [\n'
            '    {"type": "<type_string>", "raw_value": "<exact_text_substring>", '
            '"normalized_value": "<normalized_value>", "confidence": <float_score>}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- You MUST extract EVERY SINGLE occurrence of any category list above. "
            "Be extremely comprehensive and do not omit any. Check for age, gender, "
            "location, zip_code, disease, occupation, and ethnicity.\n"
            "- Extract information about PEOPLE only, not organizations or "
            "abstract concepts.\n"
            "- raw_value must be the exact substring as it appears in the "
            "text.\n"
            "- For diseases, use the standard medical name that would appear "
            "in the Orphanet rare-disease database or CDC chronic-disease "
            "database.\n"
            "- confidence is 0.0–1.0 reflecting certainty that this is a "
            "quasi-identifier about a person.\n"
            "- NEVER use or output the placeholder values from the format example (like '<type_string>'). Only output values extracted directly from the text.\n"
            '- If no quasi-identifiers are found, return '
            '{"quasi_identifiers": []}.'
        )

    # ------------------------------------------------------------------
    # Ollama communication
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str) -> str:
        """Send a prompt to the local Ollama instance and return the raw response."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "format": "json",
        }
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                try:
                    err_json = r.json()
                    err_msg = err_json.get("error", r.text)
                except Exception:
                    err_msg = r.text
                raise RuntimeError(
                    f"Ollama returned HTTP {r.status_code} for model '{self.model}': {err_msg}"
                )
            r.raise_for_status()
            return r.json().get("response", "")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Could not connect to Ollama. "
                "Make sure it is running with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s"
            )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, response: str) -> list[dict]:
        """Extract the quasi_identifiers list from the SLM JSON output."""
        response = response.strip()
        # Strip non-JSON preamble / postamble if present
        if not response.startswith("{"):
            start = response.find("{")
            if start != -1:
                response = response[start:]
        if not response.endswith("}"):
            end = response.rfind("}")
            if end != -1:
                response = response[: end + 1]
        try:
            data = json.loads(response)
            return data.get("quasi_identifiers", [])
        except json.JSONDecodeError as exc:
            log.warning("Failed to parse SLM response: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _to_quasi_identifiers(
        self, parsed: list[dict], text: str
    ) -> List[QuasiIdentifier]:
        """Convert raw SLM dicts into validated QuasiIdentifier objects."""
        qis: List[QuasiIdentifier] = []
        text_lower = text.lower()
        for item in parsed:
            qi_type_str = item.get("type", "")
            if qi_type_str not in self.VALID_QI_TYPES:
                continue

            raw = item.get("raw_value", "")
            normalized = item.get("normalized_value", raw)
            confidence = float(item.get("confidence", 0.5))

            # Locate the raw_value inside the original text for masking
            start_pos, end_pos = self._find_span(text, text_lower, raw)

            # Skip hallucinated items not present in the text
            if not raw or (start_pos == 0 and end_pos == 0 and raw.lower() not in text_lower):
                continue

            qis.append(
                QuasiIdentifier(
                    qi_type=QIType(qi_type_str),
                    raw_value=raw,
                    normalized_value=normalized.lower().strip(),
                    confidence=min(max(confidence, 0.0), 1.0),
                    start_pos=start_pos,
                    end_pos=end_pos,
                    detection_method="slm",
                )
            )
        return self._dedup(qis)

    @staticmethod
    def _find_span(
        text: str, text_lower: str, raw_value: str
    ) -> tuple[int, int]:
        """Return (start, end) character offsets for *raw_value* in *text*.

        Falls back to case-insensitive search, then to (0, 0) if not found.
        """
        idx = text.find(raw_value)
        if idx != -1:
            return idx, idx + len(raw_value)
        idx = text_lower.find(raw_value.lower())
        if idx != -1:
            return idx, idx + len(raw_value)
        return 0, 0

    @staticmethod
    def _dedup(qis: List[QuasiIdentifier]) -> List[QuasiIdentifier]:
        """Keep the highest-confidence QI per (type, normalized_value) pair."""
        best: dict[tuple, QuasiIdentifier] = {}
        for qi in qis:
            key = (qi.qi_type, qi.normalized_value)
            if key not in best or qi.confidence > best[key].confidence:
                best[key] = qi
        return list(best.values())

    # -- serialization for disk cache ----------------------------------
    @staticmethod
    def _qi_to_dict(qi: QuasiIdentifier) -> dict:
        return {
            "qi_type": qi.qi_type.value,
            "raw_value": qi.raw_value,
            "normalized_value": qi.normalized_value,
            "confidence": qi.confidence,
            "start_pos": qi.start_pos,
            "end_pos": qi.end_pos,
            "detection_method": qi.detection_method,
        }

    @staticmethod
    def _dict_to_qi(d: dict) -> QuasiIdentifier:
        return QuasiIdentifier(
            qi_type=QIType(d["qi_type"]),
            raw_value=d["raw_value"],
            normalized_value=d["normalized_value"],
            confidence=d["confidence"],
            start_pos=d["start_pos"],
            end_pos=d["end_pos"],
            detection_method=d.get("detection_method", "slm"),
        )


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
            # Querying without startyear and endyear automatically retrieves
            # the latest year available (e.g. 2025), preventing no-data errors.
            payload = {
                "seriesid": [series_id]
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
                inner_data = data.get("data", data) if isinstance(data, dict) else data
                results = inner_data.get("results", inner_data) if isinstance(inner_data, dict) else inner_data
                
                entry = None
                if isinstance(results, list) and results:
                    entry = results[0]
                elif isinstance(results, dict):
                    entry = results
                
                if entry:
                    code = entry.get("ORPHAcode", entry.get("orphacode", ""))
                    if not code and "DisorderDisorderAssociation" in entry:
                        associations = entry["DisorderDisorderAssociation"]
                        if isinstance(associations, list) and associations:
                            target = associations[0].get("TargetDisorder", {})
                            code = target.get("ORPHAcode", target.get("orphacode", ""))
                    if code:
                        return str(code)
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
            inner_data = data.get("data", data) if isinstance(data, dict) else data
            results = inner_data.get("results", inner_data) if isinstance(inner_data, dict) else inner_data
            
            prevalences = []
            if isinstance(results, dict):
                for key in ["Prevalence", "Prevalences", "prevalence", "prevalences"]:
                    if key in results:
                        prevalences = results[key]
                        break
            elif isinstance(results, list) and results:
                first = results[0]
                if isinstance(first, dict):
                    for key in ["Prevalence", "Prevalences", "prevalence", "prevalences"]:
                        if key in first:
                            prevalences = first[key]
                            break

            if not isinstance(prevalences, list):
                prevalences = [prevalences]

            for prev in prevalences:
                if not isinstance(prev, dict):
                    continue
                pclass = prev.get("PrevalenceClass", prev.get("prevalenceClass", ""))
                ptype = prev.get("PrevalenceType", prev.get("prevalenceType", ""))
                # Prefer "Point prevalence" over "Birth prevalence"
                if "point" in str(ptype).lower() or "prevalence" in str(ptype).lower():
                    for key, val in PREVALENCE_MAP.items():
                        if key.lower() in pclass.lower():
                            return val
            # If no match, try first prevalence class anyway
            if prevalences and isinstance(prevalences[0], dict):
                pclass = prevalences[0].get("PrevalenceClass", prevalences[0].get("prevalenceClass", ""))
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
        self.extractor = SLMQIExtractor()
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
        freq = self._get_frequency_by_value(qi, qi.normalized_value)
        if not freq and qi.raw_value != qi.normalized_value:
            # Fall back to raw_value in case normalization format mismatched dataset nomenclature
            freq = self._get_frequency_by_value(qi, qi.raw_value)
        return freq

    def _get_frequency_by_value(self, qi: QuasiIdentifier, value: str) -> Optional[QIFrequency]:
        if qi.qi_type in (QIType.AGE, QIType.DATE_OF_BIRTH, QIType.GENDER,
                          QIType.LOCATION, QIType.ZIP_CODE):
            return self.census.get_frequency(qi.qi_type, value)
        if qi.qi_type == QIType.DISEASE:
            return self.disease.get_frequency(qi.qi_type, value)
        if qi.qi_type == QIType.OCCUPATION:
            return self.occupation.get_frequency(qi.qi_type, value)
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