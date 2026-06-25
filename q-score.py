"""
Q-Score: Quasi-Identifier Detection and Risk Assessment
========================================================

Detects quasi-identifiers in text and calculates re-identification
risk based on k-anonymity principles using real population data.
"""

import os
import re
import json
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from functools import lru_cache
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv() 


# =============================================================================
# Data Classes and Enums
# =============================================================================

class QIType(Enum):
    """Types of quasi-identifiers we detect."""
    AGE = "age"
    DATE_OF_BIRTH = "date_of_birth"
    GENDER = "gender"
    LOCATION = "location"
    ZIP_CODE = "zip_code"
    DISEASE = "disease"
    OCCUPATION = "occupation"


@dataclass
class QuasiIdentifier:
    """Represents a detected quasi-identifier."""
    qi_type: QIType
    raw_value: str
    normalized_value: str
    confidence: float
    start_pos: int
    end_pos: int
    
    def __hash__(self):
        return hash((self.qi_type, self.normalized_value))


@dataclass
class QIFrequency:
    """Frequency/probability information for a QI value."""
    qi_type: QIType
    value: str
    probability: float
    population_count: Optional[int] = None
    source: str = "unknown"


@dataclass 
class QScoreResult:
    """Q-Score analysis result."""
    q_score: float
    expected_k: float
    detected_qis: list
    frequencies: dict
    explanation: str


# =============================================================================
# Population Data Sources
# =============================================================================

class PopulationDataSource(ABC):
    """Abstract base class for population data sources."""
    
    @abstractmethod
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        pass


class CensusDataSource(PopulationDataSource):
    """
    Fetches demographic data from the US Census Bureau API.
    
    Uses American Community Survey (ACS) 5-year estimates.
    API docs: https://www.census.gov/data/developers/data-sets.html
    """
    
    BASE_URL = "https://api.census.gov/data"
    US_POPULATION = 331_900_000
    
    STATE_FIPS = {
        "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05",
        "california": "06", "colorado": "08", "connecticut": "09", "delaware": "10",
        "florida": "12", "georgia": "13", "hawaii": "15", "idaho": "16",
        "illinois": "17", "indiana": "18", "iowa": "19", "kansas": "20",
        "kentucky": "21", "louisiana": "22", "maine": "23", "maryland": "24",
        "massachusetts": "25", "michigan": "26", "minnesota": "27", "mississippi": "28",
        "missouri": "29", "montana": "30", "nebraska": "31", "nevada": "32",
        "new hampshire": "33", "new jersey": "34", "new mexico": "35", "new york": "36",
        "north carolina": "37", "north dakota": "38", "ohio": "39", "oklahoma": "40",
        "oregon": "41", "pennsylvania": "42", "rhode island": "44", "south carolina": "45",
        "south dakota": "46", "tennessee": "47", "texas": "48", "utah": "49",
        "vermont": "50", "virginia": "51", "washington": "53", "west virginia": "54",
        "wisconsin": "55", "wyoming": "56", "district of columbia": "11"
    }
    
    # State abbreviations to full names
    STATE_ABBREV = {
        "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
        "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
        "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
        "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
        "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
        "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
        "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
        "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
        "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
        "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
        "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
        "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
        "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia"
    }
    
    # ACS table B01001: Sex by Age
    AGE_VARIABLE_MAP = {
        (0, 4): ("B01001_003E", "B01001_027E"),
        (5, 9): ("B01001_004E", "B01001_028E"),
        (10, 14): ("B01001_005E", "B01001_029E"),
        (15, 17): ("B01001_006E", "B01001_030E"),
        (18, 19): ("B01001_007E", "B01001_031E"),
        (20, 20): ("B01001_008E", "B01001_032E"),
        (21, 21): ("B01001_009E", "B01001_033E"),
        (22, 24): ("B01001_010E", "B01001_034E"),
        (25, 29): ("B01001_011E", "B01001_035E"),
        (30, 34): ("B01001_012E", "B01001_036E"),
        (35, 39): ("B01001_013E", "B01001_037E"),
        (40, 44): ("B01001_014E", "B01001_038E"),
        (45, 49): ("B01001_015E", "B01001_039E"),
        (50, 54): ("B01001_016E", "B01001_040E"),
        (55, 59): ("B01001_017E", "B01001_041E"),
        (60, 61): ("B01001_018E", "B01001_042E"),
        (62, 64): ("B01001_019E", "B01001_043E"),
        (65, 66): ("B01001_020E", "B01001_044E"),
        (67, 69): ("B01001_021E", "B01001_045E"),
        (70, 74): ("B01001_022E", "B01001_046E"),
        (75, 79): ("B01001_023E", "B01001_047E"),
        (80, 84): ("B01001_024E", "B01001_048E"),
        (85, 120): ("B01001_025E", "B01001_049E"),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CENSUS_API_KEY")
        self._age_distribution_cache: Optional[dict] = None
    
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        if qi_type == QIType.AGE:
            return self._get_age_frequency(value)
        elif qi_type == QIType.DATE_OF_BIRTH:
            # Convert DOB to age, then get age frequency
            # DOB is more identifying than age alone
            return self._get_dob_frequency(value)
        elif qi_type == QIType.GENDER:
            return self._get_gender_frequency(value)
        elif qi_type == QIType.LOCATION:
            return self._get_location_frequency(value)
        elif qi_type == QIType.ZIP_CODE:
            return self._get_zip_frequency(value)
        return None
    
    def _fetch_age_distribution(self) -> dict[int, float]:
        """Fetch real age distribution from Census ACS data."""
        if self._age_distribution_cache is not None:
            return self._age_distribution_cache
        
        all_vars = ["NAME"]
        for male_var, female_var in self.AGE_VARIABLE_MAP.values():
            all_vars.extend([male_var, female_var])
        
        try:
            url = f"{self.BASE_URL}/2022/acs/acs5"
            params = {
                "get": ",".join(all_vars),
                "for": "us:1"
            }
            if self.api_key:
                params["key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                headers = data[0]
                values = data[1]
                
                bucket_counts = {}
                for (start_age, end_age), (male_var, female_var) in self.AGE_VARIABLE_MAP.items():
                    male_idx = headers.index(male_var)
                    female_idx = headers.index(female_var)
                    total = int(values[male_idx]) + int(values[female_idx])
                    bucket_counts[(start_age, end_age)] = total
                
                total_pop = sum(bucket_counts.values())
                distribution = {}
                
                for (start_age, end_age), count in bucket_counts.items():
                    years_in_bucket = end_age - start_age + 1
                    per_year = count / years_in_bucket
                    prob_per_year = per_year / total_pop
                    
                    for age in range(start_age, min(end_age + 1, 121)):
                        distribution[age] = prob_per_year
                
                self._age_distribution_cache = distribution
                return distribution
                
        except Exception as e:
            print(f"Census API error fetching age distribution: {e}")
        
        return self._get_fallback_age_distribution()
    
    def _get_fallback_age_distribution(self) -> dict[int, float]:
        """Fallback age distribution if API fails."""
        distribution = {}
        buckets = [
            (0, 17, 0.22),
            (18, 34, 0.22),
            (35, 54, 0.25),
            (55, 74, 0.22),
            (75, 99, 0.09),
        ]
        for start, end, pct in buckets:
            years = end - start + 1
            for age in range(start, end + 1):
                distribution[age] = pct / years
        return distribution
    
    def _get_age_frequency(self, age_str: str) -> Optional[QIFrequency]:
        try:
            age = int(age_str)
        except ValueError:
            return None
        
        if age < 0 or age > 120:
            return None
        
        distribution = self._fetch_age_distribution()
        prob = distribution.get(age, distribution.get(99, 0.001))
        
        return QIFrequency(
            qi_type=QIType.AGE,
            value=str(age),
            probability=prob,
            population_count=int(prob * self.US_POPULATION),
            source="Census ACS B01001"
        )
    
    def _get_dob_frequency(self, dob_str: str) -> Optional[QIFrequency]:
        """
        Get frequency for a date of birth.
        DOB is much more identifying than age - approximately 1/365 of the age cohort.
        """
        # Parse the DOB to get age
        try:
            # Try common formats
            for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%B %d, %Y"]:
                try:
                    dob = datetime.strptime(dob_str, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                return None
            
            # Calculate age
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Get age distribution
            distribution = self._fetch_age_distribution()
            age_prob = distribution.get(age, distribution.get(99, 0.001))
            
            # DOB probability is approximately age_prob / 365
            # (assumes uniform distribution within age cohort)
            dob_prob = age_prob / 365.0
            
            return QIFrequency(
                qi_type=QIType.DATE_OF_BIRTH,
                value=dob_str,
                probability=dob_prob,
                population_count=int(dob_prob * self.US_POPULATION),
                source="Census ACS B01001 (DOB derived)"
            )
        except Exception:
            return None
    
    def _get_gender_frequency(self, gender: str) -> Optional[QIFrequency]:
        gender_lower = gender.lower().strip()
        
        gender_probs = {
            "female": 0.508, "f": 0.508, "woman": 0.508, "girl": 0.508,
            "male": 0.492, "m": 0.492, "man": 0.492, "boy": 0.492,
        }
        
        prob = gender_probs.get(gender_lower, 0.01)
        
        return QIFrequency(
            qi_type=QIType.GENDER,
            value=gender_lower,
            probability=prob,
            population_count=int(prob * self.US_POPULATION),
            source="Census Bureau"
        )
    
    @lru_cache(maxsize=256)
    def _get_location_frequency(self, location: str) -> Optional[QIFrequency]:
        location_lower = location.lower().strip()
        
        # Check if it's a state abbreviation first
        if location_lower in self.STATE_ABBREV:
            location_lower = self.STATE_ABBREV[location_lower]
        
        # Check if it's a state
        fips = self.STATE_FIPS.get(location_lower)
        if fips:
            return self._fetch_state_population(location_lower, fips)
        
        # Default for unknown locations (small city/town)
        return QIFrequency(
            qi_type=QIType.LOCATION,
            value=location,
            probability=0.0001,
            population_count=33190,
            source="Default estimate"
        )
    
    def _fetch_state_population(self, state_name: str, fips: str) -> Optional[QIFrequency]:
        try:
            url = f"{self.BASE_URL}/2022/acs/acs5"
            params = {
                "get": "B01001_001E,NAME",
                "for": f"state:{fips}"
            }
            if self.api_key:
                params["key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    population = int(data[1][0])
                    return QIFrequency(
                        qi_type=QIType.LOCATION,
                        value=state_name,
                        probability=population / self.US_POPULATION,
                        population_count=population,
                        source="Census ACS"
                    )
        except Exception as e:
            print(f"Census API error for state {state_name}: {e}")
        
        return QIFrequency(
            qi_type=QIType.LOCATION,
            value=state_name,
            probability=0.02,
            population_count=int(0.02 * self.US_POPULATION),
            source="Fallback estimate"
        )
    
    @lru_cache(maxsize=512)
    def _get_zip_frequency(self, zip_code: str) -> Optional[QIFrequency]:
        zip_clean = re.sub(r'[^0-9]', '', zip_code)[:5]
        if len(zip_clean) != 5:
            return None
        
        try:
            url = f"{self.BASE_URL}/2022/acs/acs5"
            params = {
                "get": "B01001_001E,NAME",
                "for": f"zip code tabulation area:{zip_clean}"
            }
            if self.api_key:
                params["key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    population = int(data[1][0])
                    return QIFrequency(
                        qi_type=QIType.ZIP_CODE,
                        value=zip_clean,
                        probability=population / self.US_POPULATION,
                        population_count=population,
                        source="Census ACS ZCTA"
                    )
        except Exception as e:
            print(f"Census API error for ZIP {zip_code}: {e}")
        
        return QIFrequency(
            qi_type=QIType.ZIP_CODE,
            value=zip_clean,
            probability=7500 / self.US_POPULATION,
            population_count=7500,
            source="Default ZIP estimate"
        )


class DiseaseDataSource(PopulationDataSource):
    """Disease prevalence data (Orphanet + common conditions)."""
    
    US_POPULATION = 331_900_000
    
    DISEASE_PREVALENCE = {
        # Rare diseases
        "ehlers-danlos syndrome": 1/5000, "ehlers danlos": 1/5000, "eds": 1/5000,
        "marfan syndrome": 1/5000, "marfan": 1/5000,
        "cystic fibrosis": 1/3500, "cf": 1/3500,
        "huntington disease": 1/10000, "huntington's disease": 1/10000,
        "sickle cell disease": 1/365, "sickle cell": 1/365,
        "phenylketonuria": 1/12000, "pku": 1/12000,
        "duchenne muscular dystrophy": 1/5000, "duchenne": 1/5000,
        "tay-sachs disease": 1/320000, "tay sachs": 1/320000,
        "als": 1/50000, "amyotrophic lateral sclerosis": 1/50000,
        "hemophilia": 1/5000,
        "gaucher disease": 1/40000,
        
        # Cardiac conditions
        "myocardial infarction": 1/100,  # ~3M Americans have had one
        "heart attack": 1/100,
        "acute myocardial infarction": 1/400,  # Annual incidence
        "coronary artery disease": 1/17,
        "atrial fibrillation": 1/50,
        "heart failure": 1/50,
        "cardiomyopathy": 1/500,
        
        # Common chronic conditions
        "lupus": 1/2000, "systemic lupus erythematosus": 1/2000,
        "multiple sclerosis": 1/1000, "ms": 1/1000,
        "parkinson's disease": 1/500, "parkinsons": 1/500,
        "epilepsy": 1/100,
        "diabetes": 1/10, "type 2 diabetes": 1/11, "type 1 diabetes": 1/300,
        "hypertension": 1/3, "high blood pressure": 1/3,
        "asthma": 1/13,
        "depression": 1/15,
        "anxiety": 1/5,
        "cancer": 1/200,
        "arthritis": 1/4,
        "copd": 1/25,
        "chronic obstructive pulmonary disease": 1/25,
        "stroke": 1/40,
        "alzheimer": 1/50,
        "dementia": 1/30,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ORPHANET_API_KEY")
    
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        if qi_type != QIType.DISEASE:
            return None
        
        disease_lower = value.lower().strip()
        
        # Check local database - look for partial matches
        for disease_key, prevalence in self.DISEASE_PREVALENCE.items():
            if disease_key in disease_lower or disease_lower in disease_key:
                return QIFrequency(
                    qi_type=QIType.DISEASE,
                    value=value,
                    probability=prevalence,
                    population_count=int(prevalence * self.US_POPULATION),
                    source="Orphanet/CDC prevalence data"
                )
        
        # Unknown disease - assume moderately rare
        return QIFrequency(
            qi_type=QIType.DISEASE,
            value=value,
            probability=1/100000,
            population_count=int(self.US_POPULATION / 100000),
            source="Default rare disease estimate"
        )


class OccupationDataSource(PopulationDataSource):
    """Occupation data from BLS."""
    
    TOTAL_EMPLOYED = 160_000_000
    
    OCCUPATION_EMPLOYMENT = {
        # Healthcare
        "physician": 727000, "doctor": 727000, "surgeon": 37000,
        "nurse": 3100000, "registered nurse": 3100000, "rn": 3100000,
        "pharmacist": 322000, "dentist": 155000, "therapist": 500000,
        "paramedic": 263000, "emt": 263000,
        "cardiologist": 25000,  # More specific specialty
        
        # Tech
        "software developer": 1850000, "software engineer": 1850000,
        "programmer": 1850000, "data scientist": 113000,
        "web developer": 199000,
        
        # Education
        "teacher": 4500000, "professor": 1300000,
        
        # Legal
        "lawyer": 813000, "attorney": 813000,
        
        # Business
        "accountant": 1400000, "manager": 8000000,
        "consultant": 900000,
        
        # Public safety
        "police officer": 660000, "firefighter": 330000,
        
        # Trades
        "electrician": 740000, "plumber": 480000,
        "mechanic": 775000,
        
        # Other
        "engineer": 2000000, "pilot": 135000,
        "truck driver": 2000000, "chef": 155000,
        "professional athlete": 20000, "athlete": 20000,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BLS_API_KEY")
    
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        if qi_type != QIType.OCCUPATION:
            return None
        
        occupation_lower = value.lower().strip()
        
        for occ_key, employment in self.OCCUPATION_EMPLOYMENT.items():
            if occ_key in occupation_lower or occupation_lower in occ_key:
                prob = employment / self.TOTAL_EMPLOYED
                return QIFrequency(
                    qi_type=QIType.OCCUPATION,
                    value=value,
                    probability=prob,
                    population_count=employment,
                    source="BLS Occupational Employment Statistics"
                )
        
        return QIFrequency(
            qi_type=QIType.OCCUPATION,
            value=value,
            probability=100000 / self.TOTAL_EMPLOYED,
            population_count=100000,
            source="Default occupation estimate"
        )


# =============================================================================
# Quasi-Identifier Extractor
# =============================================================================

class QIExtractor:
    """Extracts quasi-identifiers from text using pattern matching."""
    
    # Age patterns - explicit age mentions
    AGE_PATTERNS = [
        r'\b(\d{1,3})\s*[-–]?\s*(?:years?\s*old|year\s*old|yo|y\.o\.|y/o)\b',
        r'\bage[d]?\s*[:;]?\s*(\d{1,3})\b',
        r'\b(\d{1,3})\s*(?:year|yr)[\s-]*old\b',
        r'\b([1-9][0-9]?)\s*[MFmf]\b',  # "45M" or "32F" format
    ]
    
    # Date of birth patterns
    DOB_PATTERNS = [
        r'\b(?:DOB|D\.O\.B\.|Date of Birth|Birth\s*Date|Born)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(?:DOB|D\.O\.B\.|Date of Birth|Birth\s*Date|Born)[:\s]*(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\s*(?:\(DOB\)|\(Date of Birth\))',
    ]
    
    GENDER_PATTERNS = [
        r'\b(male|female|man|woman|boy|girl)\b',
        r'\bgender[:\s]*(male|female|m|f)\b',
        r'\bsex[:\s]*(male|female|m|f)\b',
    ]
    
    GENDER_MAPPING = {
        'male': 'male', 'm': 'male', 'man': 'male', 'boy': 'male',
        'female': 'female', 'f': 'female', 'woman': 'female', 'girl': 'female'
    }
    
    US_STATES = {
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
        'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
        'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
        'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
        'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
        'new hampshire', 'new jersey', 'new mexico', 'new york',
        'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
        'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
        'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
        'west virginia', 'wisconsin', 'wyoming', 'district of columbia'
    }
    
    # State abbreviations
    STATE_ABBREV = {
        'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
        'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
        'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
        'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
        'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy', 'dc'
    }
    
    STATE_ABBREV_TO_FULL = {
        'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
        'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
        'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
        'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
        'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
        'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
        'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
        'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
        'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
        'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
        'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
        'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
        'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia'
    }
    
    KNOWN_DISEASES = {
        # Cardiac
        'myocardial infarction', 'heart attack', 'coronary artery disease',
        'atrial fibrillation', 'heart failure', 'cardiomyopathy',
        # Rare
        'ehlers-danlos syndrome', 'marfan syndrome', 'cystic fibrosis',
        'huntington disease', 'sickle cell disease', 'lupus', 'diabetes',
        'hypertension', 'cancer', 'asthma', 'depression', 'anxiety',
        'multiple sclerosis', 'parkinson', 'alzheimer', 'epilepsy',
        'hemophilia', 'leukemia', 'lymphoma', 'melanoma', 'arthritis',
        'fibromyalgia', 'crohn', 'colitis', 'celiac', 'als',
        'stroke', 'copd', 'dementia',
    }
    
    # Additional disease patterns to catch ICD codes and formal diagnoses
    DISEASE_PATTERNS = [
        r'(?:diagnosis|diagnosed|dx)[:\s]*([A-Za-z\s\-]+?)(?:\s*\(|$|\n|\.)',
        r'(?:ICD-?10)[:\s]*[A-Z]\d+(?:\.\d+)?[:\s]*([A-Za-z\s\-]+)',
    ]
    
    KNOWN_OCCUPATIONS = {
        'doctor', 'physician', 'nurse', 'surgeon', 'lawyer', 'attorney',
        'teacher', 'professor', 'engineer', 'developer', 'programmer',
        'police officer', 'firefighter', 'pilot', 'chef', 'accountant',
        'dentist', 'pharmacist', 'therapist', 'scientist', 'researcher',
        'manager', 'executive', 'consultant', 'analyst', 'mechanic',
        'electrician', 'plumber', 'athlete', 'cardiologist',
    }
    
    def extract(self, text: str) -> list[QuasiIdentifier]:
        qis = []
        
        qis.extend(self._extract_ages(text))
        qis.extend(self._extract_dobs(text))
        qis.extend(self._extract_genders(text))
        qis.extend(self._extract_locations(text))
        qis.extend(self._extract_zip_codes(text))
        qis.extend(self._extract_diseases(text))
        qis.extend(self._extract_occupations(text))
        
        # Deduplicate
        seen = set()
        unique_qis = []
        for qi in qis:
            key = (qi.qi_type, qi.normalized_value)
            if key not in seen:
                seen.add(key)
                unique_qis.append(qi)
        
        return unique_qis
    
    def _extract_ages(self, text: str) -> list[QuasiIdentifier]:
        qis = []
        for pattern in self.AGE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    age = int(match.group(1))
                    if 0 <= age <= 120:
                        qis.append(QuasiIdentifier(
                            qi_type=QIType.AGE,
                            raw_value=match.group(0),
                            normalized_value=str(age),
                            confidence=0.9,
                            start_pos=match.start(),
                            end_pos=match.end()
                        ))
                except ValueError:
                    continue
        return qis
    
    def _extract_dobs(self, text: str) -> list[QuasiIdentifier]:
        """Extract dates of birth."""
        qis = []
        for pattern in self.DOB_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dob_str = match.group(1)
                qis.append(QuasiIdentifier(
                    qi_type=QIType.DATE_OF_BIRTH,
                    raw_value=match.group(0),
                    normalized_value=dob_str,
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        return qis
    
    def _extract_genders(self, text: str) -> list[QuasiIdentifier]:
        qis = []
        for pattern in self.GENDER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get the last captured group (the actual gender)
                gender_raw = match.group(match.lastindex).lower()
                if gender_raw in self.GENDER_MAPPING:
                    qis.append(QuasiIdentifier(
                        qi_type=QIType.GENDER,
                        raw_value=match.group(0),
                        normalized_value=self.GENDER_MAPPING[gender_raw],
                        confidence=0.95,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        return qis
    
    def _extract_locations(self, text: str) -> list[QuasiIdentifier]:
        qis = []
        
        # Check for full state names
        for state in self.US_STATES:
            pattern = r'\b' + re.escape(state) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                qis.append(QuasiIdentifier(
                    qi_type=QIType.LOCATION,
                    raw_value=match.group(0),
                    normalized_value=state.title(),
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        # Check for state abbreviations (with word boundaries and context)
        # Pattern: comma + space + abbreviation, or abbreviation + ZIP
        abbrev_pattern = r'(?:,\s*|\b)([A-Z]{2})\s*(?:\d{5}|$|\n)'
        for match in re.finditer(abbrev_pattern, text):
            abbrev = match.group(1).lower()
            if abbrev in self.STATE_ABBREV:
                full_name = self.STATE_ABBREV_TO_FULL[abbrev]
                qis.append(QuasiIdentifier(
                    qi_type=QIType.LOCATION,
                    raw_value=match.group(1),
                    normalized_value=full_name,
                    confidence=0.8,
                    start_pos=match.start(1),
                    end_pos=match.end(1)
                ))
        
        return qis
    
    def _extract_zip_codes(self, text: str) -> list[QuasiIdentifier]:
        qis = []
        
        # Pattern 1: ZIP with context words nearby
        context_pattern = r'(?:zip|postal|address|located|lives|resides)[^0-9]{0,30}(\d{5})(?:-\d{4})?'
        for match in re.finditer(context_pattern, text, re.IGNORECASE):
            qis.append(QuasiIdentifier(
                qi_type=QIType.ZIP_CODE,
                raw_value=match.group(1),
                normalized_value=match.group(1),
                confidence=0.95,
                start_pos=match.start(1),
                end_pos=match.end(1)
            ))
        
        # Pattern 2: State abbreviation followed by ZIP
        state_zip_pattern = r'\b[A-Z]{2}\s+(\d{5})(?:-\d{4})?\b'
        for match in re.finditer(state_zip_pattern, text):
            zip_code = match.group(1)
            # Verify it's not already captured
            already_found = any(qi.normalized_value == zip_code for qi in qis)
            if not already_found:
                qis.append(QuasiIdentifier(
                    qi_type=QIType.ZIP_CODE,
                    raw_value=match.group(0),
                    normalized_value=zip_code,
                    confidence=0.9,
                    start_pos=match.start(1),
                    end_pos=match.end(1)
                ))
        
        return qis
    
    def _extract_diseases(self, text: str) -> list[QuasiIdentifier]:
        qis = []
        text_lower = text.lower()
        
        # Check for known diseases
        for disease in self.KNOWN_DISEASES:
            if disease in text_lower:
                idx = text_lower.find(disease)
                qis.append(QuasiIdentifier(
                    qi_type=QIType.DISEASE,
                    raw_value=text[idx:idx+len(disease)],
                    normalized_value=disease,
                    confidence=0.95,
                    start_pos=idx,
                    end_pos=idx + len(disease)
                ))
        
        # Try to extract from diagnostic patterns
        for pattern in self.DISEASE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                disease = match.group(1).strip().lower()
                # Filter out very short or obviously non-disease strings
                if len(disease) > 4 and disease not in [d.normalized_value for d in qis]:
                    qis.append(QuasiIdentifier(
                        qi_type=QIType.DISEASE,
                        raw_value=match.group(0),
                        normalized_value=disease,
                        confidence=0.7,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return qis
    
    def _extract_occupations(self, text: str) -> list[QuasiIdentifier]:
        qis = []
        text_lower = text.lower()
        
        for occupation in self.KNOWN_OCCUPATIONS:
            if occupation in text_lower:
                idx = text_lower.find(occupation)
                qis.append(QuasiIdentifier(
                    qi_type=QIType.OCCUPATION,
                    raw_value=text[idx:idx+len(occupation)],
                    normalized_value=occupation,
                    confidence=0.85,
                    start_pos=idx,
                    end_pos=idx + len(occupation)
                ))
        
        return qis


# =============================================================================
# Q-Score Calculator
# =============================================================================

class QScoreCalculator:
    """Calculates Q-Score based on k-anonymity principles."""
    
    REFERENCE_POPULATION = 331_900_000
    
    def __init__(self, k_threshold: int = 5):
        self.census = CensusDataSource()
        self.disease = DiseaseDataSource()
        self.occupation = OccupationDataSource()
        self.extractor = QIExtractor()
        self.k_threshold = k_threshold
    
    def calculate(self, text: str) -> QScoreResult:
        # Extract QIs
        detected_qis = self.extractor.extract(text)
        
        if not detected_qis:
            return QScoreResult(
                q_score=0.0,
                expected_k=self.REFERENCE_POPULATION,
                detected_qis=[],
                frequencies={},
                explanation="No quasi-identifiers detected."
            )
        
        # Look up frequencies
        frequencies = {}
        for qi in detected_qis:
            freq = self._get_frequency(qi)
            if freq:
                frequencies[qi] = freq
        
        if not frequencies:
            return QScoreResult(
                q_score=0.0,
                expected_k=self.REFERENCE_POPULATION,
                detected_qis=detected_qis,
                frequencies={},
                explanation=f"Detected {len(detected_qis)} QI(s) but could not lookup frequencies."
            )
        
        # Calculate joint probability (independence assumption)
        joint_prob = 1.0
        for freq in frequencies.values():
            joint_prob *= freq.probability
        
        # Expected equivalence class size
        expected_k = self.REFERENCE_POPULATION * joint_prob
        
        # Convert to Q-Score
        if expected_k >= self.k_threshold:
            q_score = 0.0
        else:
            q_score = min(1.0, 1.0 - (expected_k / self.k_threshold))
        
        # Generate explanation
        explanation = self._generate_explanation(detected_qis, frequencies, expected_k)
        
        return QScoreResult(
            q_score=q_score,
            expected_k=expected_k,
            detected_qis=detected_qis,
            frequencies=frequencies,
            explanation=explanation
        )
    
    def _get_frequency(self, qi: QuasiIdentifier) -> Optional[QIFrequency]:
        if qi.qi_type in [QIType.AGE, QIType.DATE_OF_BIRTH, QIType.GENDER, QIType.LOCATION, QIType.ZIP_CODE]:
            return self.census.get_frequency(qi.qi_type, qi.normalized_value)
        elif qi.qi_type == QIType.DISEASE:
            return self.disease.get_frequency(qi.qi_type, qi.normalized_value)
        elif qi.qi_type == QIType.OCCUPATION:
            return self.occupation.get_frequency(qi.qi_type, qi.normalized_value)
        return None
    
    def _generate_explanation(
        self, 
        detected_qis: list, 
        frequencies: dict, 
        expected_k: float
    ) -> str:
        lines = [f"Detected {len(detected_qis)} quasi-identifier(s):"]
        
        for qi in detected_qis:
            freq = frequencies.get(qi)
            if freq:
                lines.append(
                    f"  - {qi.qi_type.value}: '{qi.normalized_value}' "
                    f"(P ≈ {freq.probability:.2e}, ~{freq.population_count:,} people)"
                )
            else:
                lines.append(
                    f"  - {qi.qi_type.value}: '{qi.normalized_value}' (frequency unknown)"
                )
        
        lines.append(f"\nExpected equivalence class size E[k]: {expected_k:.4f}")
        
        if expected_k < 1:
            lines.append("🚨 E[k] < 1: Likely UNIQUELY IDENTIFYING")
        elif expected_k < self.k_threshold:
            lines.append(f"⚠️  E[k] < {self.k_threshold}: HIGH re-identification risk")
        else:
            lines.append(f"✓ E[k] ≥ {self.k_threshold}: Acceptable k-anonymity")
        
        return "\n".join(lines)


# =============================================================================
# Main Interface
# =============================================================================

class QScoreAnalyzer:
    """Main interface for Q-Score analysis."""
    
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
            "risk_level": "HIGH" if result.q_score >= 0.7 else "MEDIUM" if result.q_score >= 0.3 else "LOW",
            "k_threshold": self.calculator.k_threshold,
            "detected_quasi_identifiers": [
                {
                    "type": qi.qi_type.value,
                    "raw_value": qi.raw_value,
                    "normalized_value": qi.normalized_value,
                    "confidence": qi.confidence,
                    "position": {"start": qi.start_pos, "end": qi.end_pos},
                    "frequency": {
                        "probability": result.frequencies[qi].probability,
                        "population_count": result.frequencies[qi].population_count,
                        "source": result.frequencies[qi].source
                    } if qi in result.frequencies else None
                }
                for qi in result.detected_qis
            ],
            "explanation": result.explanation
        }


# =============================================================================
# Public API
# =============================================================================

def calculate_qscore(text: str) -> dict:
    """
    Calculate Q-Score for text.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary with Q-Score analysis
    """
    analyzer = QScoreAnalyzer()
    return analyzer.get_report(text)


if __name__ == "__main__":
    sample = """
    Patient: John Michael Smith
    DOB: 03/15/1978
    SSN: 123-45-6789
    MRN: MR-12345678
    Address: 456 Oak Street, Millbrook, NY 12545
    Phone: (845) 555-1234
    Email: john.smith@email.com

    Chief Complaint: Patient presents with chest pain and shortness of breath.
    Diagnosis: Acute myocardial infarction (ICD-10: I21.9)
    Attending Physician: Dr. Williams

    The patient was started on aspirin 325mg and atorvastatin 40mg.
    Follow-up scheduled at Mayo Clinic Cardiology department.
    """
    
    report = calculate_qscore(sample)
    print(json.dumps(report, indent=2))