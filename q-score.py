"""
Q-Score: Quasi-Identifier Detection and Risk Assessment
========================================================

This module detects quasi-identifiers in text and calculates re-identification
risk based on k-anonymity principles using real population data.
"""

import re
import json
import math
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from functools import lru_cache
from itertools import combinations
from datetime import datetime
from pathlib import Path

from example_inputs import load_example_inputs


# =============================================================================
# Data Classes and Enums
# =============================================================================

class QIType(Enum):
    """Types of quasi-identifiers we detect."""
    AGE = "age"
    GENDER = "gender"
    LOCATION = "location"  # State/city
    ZIP_CODE = "zip_code"
    DISEASE = "disease"
    OCCUPATION = "occupation"
    ETHNICITY = "ethnicity"
    MARITAL_STATUS = "marital_status"


@dataclass
class QuasiIdentifier:
    """Represents a detected quasi-identifier."""
    qi_type: QIType
    raw_value: str  # Original text
    normalized_value: str  # Standardized value for lookup
    confidence: float  # Detection confidence (0-1)
    start_pos: int  # Position in text
    end_pos: int
    
    def __hash__(self):
        return hash((self.qi_type, self.normalized_value))


@dataclass
class QIFrequency:
    """Frequency/probability information for a QI value."""
    qi_type: QIType
    value: str
    probability: float  # π(v) - proportion in population
    population_count: Optional[int] = None  # Absolute count if available
    source: str = "unknown"
    is_estimated: bool = False  # True if using fallback/estimation


@dataclass
class QScoreResult:
    """Complete Q-Score analysis result."""
    q_score: float  # Final Q-score (0-1)
    q_rarest: float  # Risk from rarest single QI
    q_combination: float  # Risk from full combination
    q_subsets: float  # Risk from identifying subsets
    
    detected_qis: list  # List of QuasiIdentifier
    frequencies: dict  # QI -> QIFrequency mapping
    expected_k: float  # Expected equivalence class size
    identifying_subsets: list  # Subsets with E[k] < threshold
    
    explanation: str  # Human-readable explanation
    
    # Weights used
    weights: dict = field(default_factory=lambda: {
        "w1_rarest": 0.3,
        "w2_combination": 0.5,
        "w3_subsets": 0.2
    })


# =============================================================================
# Abstract Base Class for Data Sources
# =============================================================================

class PopulationDataSource(ABC):
    """Abstract base class for population data sources."""
    
    @abstractmethod
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        """Get the population frequency for a QI value."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the data source is available."""
        pass


# =============================================================================
# Census Bureau Data Source
# =============================================================================

class CensusDataSource(PopulationDataSource):
    """
    Fetches demographic data from the US Census Bureau API.
    
    API Documentation: https://www.census.gov/data/developers/data-sets.html
    """
    
    BASE_URL = "https://api.census.gov/data"
    US_POPULATION = 331_900_000  # 2023 estimate
    
    # State FIPS codes
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
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Census data source.
        
        Args:
            api_key: Census API key (optional but recommended for higher rate limits)
                     Get one at: https://api.census.gov/data/key_signup.html
        """
        self.api_key = api_key
        self._cache = {}
    
    def is_available(self) -> bool:
        """Check if Census API is reachable."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/2021/acs/acs5?get=NAME&for=state:01",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        """Get frequency for age, gender, location, or ZIP code."""
        if qi_type == QIType.AGE:
            return self._get_age_frequency(value)
        elif qi_type == QIType.GENDER:
            return self._get_gender_frequency(value)
        elif qi_type == QIType.LOCATION:
            return self._get_location_frequency(value)
        elif qi_type == QIType.ZIP_CODE:
            return self._get_zip_frequency(value)
        return None
    
    @lru_cache(maxsize=128)
    def _get_age_frequency(self, age_str: str) -> Optional[QIFrequency]:
        """
        Get frequency for a specific age.
        Uses ACS 5-year estimates for age distribution.
        """
        try:
            age = int(age_str)
        except ValueError:
            return None
        
        # Age distribution approximation based on Census data
        # These are approximate probabilities for single-year ages
        age_probabilities = self._get_age_distribution()
        
        if age < 0 or age > 100:
            return None
        
        # Clamp to available range
        lookup_age = min(age, 99)
        prob = age_probabilities.get(lookup_age, 0.005)  # Default for very old ages
        
        return QIFrequency(
            qi_type=QIType.AGE,
            value=str(age),
            probability=prob,
            population_count=int(prob * self.US_POPULATION),
            source="Census ACS 5-year estimates (approximated)",
            is_estimated=True
        )
    
    def _get_age_distribution(self) -> dict:
        """
        Returns approximate single-year age probabilities.
        Based on Census Bureau age distribution data.
        """
        # Simplified age distribution (would be fetched from API in production)
        # This represents P(age = x) for the US population
        distribution = {}
        
        # Children (0-17): ~22% of population
        for age in range(0, 18):
            distribution[age] = 0.22 / 18  # ~0.0122 each
        
        # Young adults (18-34): ~22% of population  
        for age in range(18, 35):
            distribution[age] = 0.22 / 17  # ~0.0129 each
        
        # Middle age (35-54): ~25% of population
        for age in range(35, 55):
            distribution[age] = 0.25 / 20  # ~0.0125 each
        
        # Older adults (55-74): ~22% of population
        for age in range(55, 75):
            distribution[age] = 0.22 / 20  # ~0.011 each
        
        # Elderly (75+): ~9% of population
        for age in range(75, 100):
            base_prob = 0.09 / 25
            # Decreasing probability with age
            distribution[age] = base_prob * (0.95 ** (age - 75))
        
        return distribution
    
    @lru_cache(maxsize=8)
    def _get_gender_frequency(self, gender: str) -> Optional[QIFrequency]:
        """Get frequency for gender."""
        gender_lower = gender.lower().strip()
        
        # Based on Census data: ~51% female, ~49% male
        gender_probs = {
            "female": 0.508,
            "f": 0.508,
            "woman": 0.508,
            "male": 0.492,
            "m": 0.492,
            "man": 0.492,
        }
        
        prob = gender_probs.get(gender_lower)
        if prob is None:
            # Non-binary/other - much smaller population
            prob = 0.01  # Rough estimate
        
        return QIFrequency(
            qi_type=QIType.GENDER,
            value=gender_lower,
            probability=prob,
            population_count=int(prob * self.US_POPULATION),
            source="Census Bureau estimates",
            is_estimated=False
        )
    
    @lru_cache(maxsize=256)
    def _get_location_frequency(self, location: str) -> Optional[QIFrequency]:
        """
        Get frequency for a location (state or city).
        """
        location_lower = location.lower().strip()
        
        # Check if it's a state
        if location_lower in self.STATE_FIPS:
            return self._get_state_population(location_lower)
        
        # Try to fetch city population via API
        return self._get_city_population(location)
    
    def _get_state_population(self, state_name: str) -> Optional[QIFrequency]:
        """Fetch state population from Census API."""
        fips = self.STATE_FIPS.get(state_name.lower())
        if not fips:
            return None
        
        try:
            url = f"{self.BASE_URL}/2021/acs/acs5"
            params = {
                "get": "B01001_001E,NAME",  # Total population
                "for": f"state:{fips}"
            }
            if self.api_key:
                params["key"] = self.api_key
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Response format: [["B01001_001E", "NAME", "state"], ["12345", "State Name", "01"]]
                if len(data) > 1:
                    population = int(data[1][0])
                    prob = population / self.US_POPULATION
                    
                    return QIFrequency(
                        qi_type=QIType.LOCATION,
                        value=state_name,
                        probability=prob,
                        population_count=population,
                        source="Census ACS 5-year estimates",
                        is_estimated=False
                    )
        except Exception as e:
            print(f"Census API error for state {state_name}: {e}")
        
        # Fallback to estimates
        return self._get_state_population_fallback(state_name)
    
    def _get_state_population_fallback(self, state_name: str) -> Optional[QIFrequency]:
        """Fallback state populations (2023 estimates)."""
        # Top states by population for fallback
        state_pops = {
            "california": 39_538_223,
            "texas": 29_145_505,
            "florida": 21_538_187,
            "new york": 20_201_249,
            "pennsylvania": 13_002_700,
            "illinois": 12_812_508,
            "ohio": 11_799_448,
            "georgia": 10_711_908,
            "north carolina": 10_439_388,
            "michigan": 10_077_331,
            # ... would include all states in production
        }
        
        pop = state_pops.get(state_name.lower())
        if pop:
            return QIFrequency(
                qi_type=QIType.LOCATION,
                value=state_name,
                probability=pop / self.US_POPULATION,
                population_count=pop,
                source="Census Bureau estimates (cached)",
                is_estimated=True
            )
        
        # Default for unknown locations
        return QIFrequency(
            qi_type=QIType.LOCATION,
            value=state_name,
            probability=0.01,  # Assume small
            population_count=int(0.01 * self.US_POPULATION),
            source="Default estimate",
            is_estimated=True
        )
    
    def _get_city_population(self, city_name: str) -> Optional[QIFrequency]:
        """
        Attempt to get city population.
        In production, would use Census Place API.
        """
        # Common cities fallback
        city_pops = {
            "new york city": 8_336_817,
            "los angeles": 3_979_576,
            "chicago": 2_693_976,
            "houston": 2_320_268,
            "phoenix": 1_680_992,
            "philadelphia": 1_584_064,
            "san antonio": 1_547_253,
            "san diego": 1_423_851,
            "dallas": 1_343_573,
            "san jose": 1_021_795,
        }
        
        pop = city_pops.get(city_name.lower())
        if pop:
            return QIFrequency(
                qi_type=QIType.LOCATION,
                value=city_name,
                probability=pop / self.US_POPULATION,
                population_count=pop,
                source="Census Bureau estimates (cached)",
                is_estimated=True
            )
        
        # For small/unknown cities, assume small population
        return QIFrequency(
            qi_type=QIType.LOCATION,
            value=city_name,
            probability=0.00001,  # ~3,300 people
            population_count=3300,
            source="Default small city estimate",
            is_estimated=True
        )
    
    @lru_cache(maxsize=512)
    def _get_zip_frequency(self, zip_code: str) -> Optional[QIFrequency]:
        """
        Get population for a ZIP code.
        Uses ZCTA (ZIP Code Tabulation Area) data.
        """
        # Clean ZIP code
        zip_clean = re.sub(r'[^0-9]', '', zip_code)[:5]
        if len(zip_clean) != 5:
            return None
        
        try:
            url = f"{self.BASE_URL}/2021/acs/acs5"
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
                    prob = population / self.US_POPULATION
                    
                    return QIFrequency(
                        qi_type=QIType.ZIP_CODE,
                        value=zip_clean,
                        probability=prob,
                        population_count=population,
                        source="Census ACS ZCTA data",
                        is_estimated=False
                    )
        except Exception as e:
            print(f"Census API error for ZIP {zip_code}: {e}")
        
        # Fallback: average ZIP code population
        avg_zip_pop = 7500  # Rough average
        return QIFrequency(
            qi_type=QIType.ZIP_CODE,
            value=zip_clean,
            probability=avg_zip_pop / self.US_POPULATION,
            population_count=avg_zip_pop,
            source="Default ZIP estimate",
            is_estimated=True
        )


# =============================================================================
# Orphanet Data Source (Disease Prevalence)
# =============================================================================

class OrphanetDataSource(PopulationDataSource):
    """
    Fetches rare disease prevalence data from Orphanet.
    
    Orphanet is the reference portal for rare diseases and orphan drugs.
    API Documentation: https://www.orpha.net/consor/cgi-bin/index.php
    """
    
    # Orphanet API endpoint (using their nomenclature API)
    BASE_URL = "https://api.orphacode.org"
    
    # Fallback prevalence data for common rare diseases
    # Prevalence expressed as proportion of population
    DISEASE_PREVALENCE = {
        # Connective tissue disorders
        "ehlers-danlos syndrome": 1 / 5000,
        "ehlers danlos": 1 / 5000,
        "eds": 1 / 5000,
        "marfan syndrome": 1 / 5000,
        "marfan": 1 / 5000,
        
        # Genetic disorders
        "cystic fibrosis": 1 / 3500,
        "cf": 1 / 3500,
        "huntington disease": 1 / 10000,
        "huntington's disease": 1 / 10000,
        "huntingtons": 1 / 10000,
        "sickle cell disease": 1 / 365,  # In US, varies by population
        "sickle cell": 1 / 365,
        "phenylketonuria": 1 / 12000,
        "pku": 1 / 12000,
        "duchenne muscular dystrophy": 1 / 5000,
        "duchenne": 1 / 5000,
        "fragile x syndrome": 1 / 4000,
        "fragile x": 1 / 4000,
        "tay-sachs disease": 1 / 320000,
        "tay sachs": 1 / 320000,
        
        # Autoimmune
        "lupus": 1 / 2000,
        "systemic lupus erythematosus": 1 / 2000,
        "sle": 1 / 2000,
        "multiple sclerosis": 1 / 1000,
        "ms": 1 / 1000,
        "myasthenia gravis": 1 / 5000,
        
        # Neurological
        "als": 1 / 50000,
        "amyotrophic lateral sclerosis": 1 / 50000,
        "lou gehrig's disease": 1 / 50000,
        "parkinson's disease": 1 / 500,  # Age-dependent
        "parkinsons": 1 / 500,
        
        # Rare cancers
        "mesothelioma": 1 / 100000,
        "glioblastoma": 1 / 33000,
        
        # Blood disorders
        "hemophilia": 1 / 5000,
        "hemophilia a": 1 / 5000,
        "hemophilia b": 1 / 25000,
        "von willebrand disease": 1 / 100,  # Most common bleeding disorder
        
        # Metabolic
        "gaucher disease": 1 / 40000,
        "fabry disease": 1 / 40000,
        "pompe disease": 1 / 40000,
        
        # Common chronic diseases (for contrast)
        "diabetes": 1 / 10,
        "type 2 diabetes": 1 / 11,
        "type 1 diabetes": 1 / 300,
        "hypertension": 1 / 3,
        "high blood pressure": 1 / 3,
        "asthma": 1 / 13,
        "depression": 1 / 15,
        "anxiety": 1 / 5,
        "cancer": 1 / 200,  # Annual incidence
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Orphanet data source.
        
        Args:
            api_key: Orphanet API key if required
        """
        self.api_key = api_key
    
    def is_available(self) -> bool:
        """Check if Orphanet API is reachable."""
        # For now, always return True since we have fallback data
        return True
    
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        """Get prevalence for a disease."""
        if qi_type != QIType.DISEASE:
            return None
        
        return self._get_disease_prevalence(value)
    
    def _get_disease_prevalence(self, disease_name: str) -> Optional[QIFrequency]:
        """
        Get prevalence for a disease.
        First tries Orphanet API, then falls back to local data.
        """
        disease_lower = disease_name.lower().strip()
        
        # Try API first (if implemented)
        api_result = self._query_orphanet_api(disease_lower)
        if api_result:
            return api_result
        
        # Check local database
        for disease_key, prevalence in self.DISEASE_PREVALENCE.items():
            if disease_key in disease_lower or disease_lower in disease_key:
                return QIFrequency(
                    qi_type=QIType.DISEASE,
                    value=disease_name,
                    probability=prevalence,
                    population_count=int(prevalence * 331_900_000),
                    source="Orphanet prevalence data (cached)",
                    is_estimated=True
                )
        
        # Unknown disease - assume rare
        # Default to 1 in 100,000 for unknown diseases mentioned in medical context
        default_prevalence = 1 / 100000
        return QIFrequency(
            qi_type=QIType.DISEASE,
            value=disease_name,
            probability=default_prevalence,
            population_count=int(default_prevalence * 331_900_000),
            source="Default rare disease estimate",
            is_estimated=True
        )
    
    def _query_orphanet_api(self, disease_name: str) -> Optional[QIFrequency]:
        """
        Query Orphanet API for disease prevalence.
        
        Note: Orphanet's API requires registration and has specific endpoints.
        This is a placeholder for the actual implementation.
        """
        # Orphanet API implementation would go here
        # The API provides:
        # - Disease search by name
        # - Prevalence class (e.g., 1-9/100,000)
        # - Point prevalence when available
        
        # Example API call structure:
        # try:
        #     response = requests.get(
        #         f"{self.BASE_URL}/EN/ClinicalEntity/search",
        #         params={"name": disease_name},
        #         headers={"Authorization": f"Bearer {self.api_key}"},
        #         timeout=10
        #     )
        #     if response.status_code == 200:
        #         data = response.json()
        #         # Parse prevalence from response
        #         ...
        # except Exception:
        #     pass
        
        return None  # Fall back to local data


# =============================================================================
# Bureau of Labor Statistics Data Source
# =============================================================================

class BLSDataSource(PopulationDataSource):
    """
    Fetches occupation data from Bureau of Labor Statistics.
    
    API Documentation: https://www.bls.gov/developers/
    """
    
    BASE_URL = "https://api.bls.gov/publicAPI/v2"
    TOTAL_EMPLOYED = 160_000_000  # Approximate US employed population
    
    # Occupation employment data (SOC codes and employment levels)
    # Source: BLS Occupational Employment and Wage Statistics
    OCCUPATION_EMPLOYMENT = {
        # Healthcare
        "physician": 727000,
        "doctor": 727000,
        "surgeon": 37000,
        "nurse": 3100000,
        "registered nurse": 3100000,
        "rn": 3100000,
        "nurse practitioner": 234000,
        "pharmacist": 322000,
        "dentist": 155000,
        "physical therapist": 258000,
        "occupational therapist": 137000,
        "psychologist": 192000,
        "psychiatrist": 37000,
        "paramedic": 263000,
        "emt": 263000,
        "medical assistant": 743000,
        
        # Technology
        "software developer": 1850000,
        "software engineer": 1850000,
        "programmer": 1850000,
        "data scientist": 113000,
        "web developer": 199000,
        "network administrator": 349000,
        "cybersecurity analyst": 163000,
        "it manager": 482000,
        "database administrator": 168000,
        
        # Education
        "teacher": 4500000,
        "elementary school teacher": 1400000,
        "high school teacher": 1000000,
        "professor": 1300000,
        "college professor": 1300000,
        "principal": 300000,
        
        # Legal
        "lawyer": 813000,
        "attorney": 813000,
        "paralegal": 345000,
        "judge": 28000,
        
        # Business/Finance
        "accountant": 1400000,
        "cpa": 1400000,
        "financial analyst": 303000,
        "manager": 8000000,
        "ceo": 200000,
        "executive": 2700000,
        "consultant": 900000,
        "marketing manager": 316000,
        
        # Public Safety
        "police officer": 660000,
        "cop": 660000,
        "firefighter": 330000,
        "detective": 110000,
        
        # Skilled Trades
        "electrician": 740000,
        "plumber": 480000,
        "carpenter": 1000000,
        "mechanic": 775000,
        "welder": 420000,
        
        # Service
        "chef": 155000,
        "cook": 2500000,
        "waiter": 2100000,
        "waitress": 2100000,
        "server": 2100000,
        "bartender": 650000,
        
        # Transportation
        "truck driver": 2000000,
        "pilot": 135000,
        "airline pilot": 135000,
        
        # Other
        "engineer": 2000000,
        "scientist": 800000,
        "researcher": 800000,
        "journalist": 50000,
        "writer": 150000,
        "artist": 90000,
        "musician": 170000,
        "athlete": 20000,
        "professional athlete": 20000,
        "farmer": 970000,
        "construction worker": 1500000,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize BLS data source.
        
        Args:
            api_key: BLS API key (optional, allows more queries)
                     Get one at: https://data.bls.gov/registrationEngine/
        """
        self.api_key = api_key
    
    def is_available(self) -> bool:
        """Check if BLS API is reachable."""
        return True  # We have fallback data
    
    def get_frequency(self, qi_type: QIType, value: str) -> Optional[QIFrequency]:
        """Get employment frequency for an occupation."""
        if qi_type != QIType.OCCUPATION:
            return None
        
        return self._get_occupation_frequency(value)
    
    def _get_occupation_frequency(self, occupation: str) -> Optional[QIFrequency]:
        """Get frequency for an occupation."""
        occupation_lower = occupation.lower().strip()
        
        # Search for matching occupation
        for occ_key, employment in self.OCCUPATION_EMPLOYMENT.items():
            if occ_key in occupation_lower or occupation_lower in occ_key:
                prob = employment / self.TOTAL_EMPLOYED
                return QIFrequency(
                    qi_type=QIType.OCCUPATION,
                    value=occupation,
                    probability=prob,
                    population_count=employment,
                    source="BLS Occupational Employment Statistics",
                    is_estimated=True
                )
        
        # Unknown occupation - assume moderately common
        default_employment = 100000
        return QIFrequency(
            qi_type=QIType.OCCUPATION,
            value=occupation,
            probability=default_employment / self.TOTAL_EMPLOYED,
            population_count=default_employment,
            source="Default occupation estimate",
            is_estimated=True
        )


# =============================================================================
# Quasi-Identifier Extractor
# =============================================================================

class QIExtractor:
    """
    Extracts quasi-identifiers from text using pattern matching and NLP.
    """
    
    # Age patterns
    AGE_PATTERNS = [
        r'\b(\d{1,3})\s*(?:years?\s*old|year\s*old|yo|y\.o\.|y/o)\b',
        r'\bage[d]?\s*(\d{1,3})\b',
        r'\b(\d{1,3})\s*(?:year|yr)[\s-]*old\b',
        r'\bis\s*(\d{1,3})\b(?=.*\b(?:patient|male|female|man|woman|person)\b)',
    ]
    
    # Gender patterns
    GENDER_PATTERNS = [
        r'\b(male|female|man|woman|boy|girl)\b',
        r'\b(m|f)\b(?=\s*/\s*\d)',  # M/45, F/32
        r'\b(\d+)\s*(m|f)\b',  # 45M, 32F
    ]
    
    GENDER_MAPPING = {
        'male': 'male', 'm': 'male', 'man': 'male', 'boy': 'male',
        'female': 'female', 'f': 'female', 'woman': 'female', 'girl': 'female'
    }
    
    # Location patterns
    LOCATION_PATTERNS = [
        # State names will be matched against a list
        r'\b(?:lives?\s+in|from|resides?\s+in|resident\s+of|located\s+in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*(?:USA|US|United States)',
    ]
    
    # ZIP code pattern
    ZIP_PATTERNS = [
        r'\b(\d{5})(?:-\d{4})?\b',
    ]
    
    # Disease patterns (simplified - would use medical NER in production)
    DISEASE_PATTERNS = [
        r'\bdiagnosed\s+with\s+([A-Za-z\s\-\']+?)(?:\.|,|\band\b|$)',
        r'\bhas\s+([A-Za-z\s\-\']+?(?:disease|syndrome|disorder|cancer|oma))',
        r'\bsuffers?\s+from\s+([A-Za-z\s\-\']+?)(?:\.|,|\band\b|$)',
        r'\bpatient\s+with\s+([A-Za-z\s\-\']+?)(?:\.|,|\band\b|$)',
        r'\b([\w\s\-\']+?(?:disease|syndrome|disorder|cancer|\'s\s+disease))\b',
    ]
    
    # Occupation patterns
    OCCUPATION_PATTERNS = [
        r'\bworks?\s+as\s+(?:a|an)?\s*([A-Za-z\s]+?)(?:\.|,|\bat\b|$)',
        r'\bis\s+(?:a|an)\s+([A-Za-z\s]+?)(?:\.|,|\bat\b|\bwho\b|$)',
        r'\b([A-Za-z]+)\s+(?:at|for)\s+(?:a|an|the)\s+(?:hospital|clinic|firm|company)',
        r'\boccupation[:\s]+([A-Za-z\s]+?)(?:\.|,|$)',
        r'\bprofession[:\s]+([A-Za-z\s]+?)(?:\.|,|$)',
    ]
    
    # Known US states for validation
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
    
    # Known diseases for validation
    KNOWN_DISEASES = {
        'ehlers-danlos syndrome', 'marfan syndrome', 'cystic fibrosis',
        'huntington disease', 'sickle cell disease', 'lupus', 'diabetes',
        'hypertension', 'cancer', 'asthma', 'depression', 'anxiety',
        'multiple sclerosis', 'parkinson', 'alzheimer', 'epilepsy',
        'hemophilia', 'leukemia', 'lymphoma', 'melanoma', 'arthritis',
        'fibromyalgia', 'crohn', 'colitis', 'celiac'
    }
    
    def extract(self, text: str) -> list[QuasiIdentifier]:
        """
        Extract all quasi-identifiers from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected QuasiIdentifier objects
        """
        qis = []
        text_lower = text.lower()
        
        # Extract each type
        qis.extend(self._extract_ages(text))
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
        """Extract age mentions."""
        qis = []
        for pattern in self.AGE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                age_str = match.group(1)
                try:
                    age = int(age_str)
                    if 0 <= age <= 120:  # Valid age range
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
    
    def _extract_genders(self, text: str) -> list[QuasiIdentifier]:
        """Extract gender mentions."""
        qis = []
        for pattern in self.GENDER_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get the gender capture group (might be in different positions)
                gender_raw = None
                for g in match.groups():
                    if g and g.lower() in self.GENDER_MAPPING:
                        gender_raw = g
                        break
                
                if gender_raw:
                    normalized = self.GENDER_MAPPING[gender_raw.lower()]
                    qis.append(QuasiIdentifier(
                        qi_type=QIType.GENDER,
                        raw_value=match.group(0),
                        normalized_value=normalized,
                        confidence=0.95,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        return qis
    
    def _extract_locations(self, text: str) -> list[QuasiIdentifier]:
        """Extract location mentions."""
        qis = []
        
        # Check for state names directly
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
        
        # Also check for "town of X" or "city of X" patterns
        town_pattern = r'\b(?:town|city|village)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        for match in re.finditer(town_pattern, text):
            location = match.group(1)
            qis.append(QuasiIdentifier(
                qi_type=QIType.LOCATION,
                raw_value=match.group(0),
                normalized_value=location,
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Check for population mentions (e.g., "town of 1,500")
        pop_pattern = r'\b(?:town|city|village)\s+of\s+([\d,]+)\s*(?:people|residents)?'
        for match in re.finditer(pop_pattern, text, re.IGNORECASE):
            pop_str = match.group(1).replace(',', '')
            try:
                population = int(pop_str)
                # Store as a special location indicator
                qis.append(QuasiIdentifier(
                    qi_type=QIType.LOCATION,
                    raw_value=match.group(0),
                    normalized_value=f"town_pop_{population}",
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
            except ValueError:
                continue
        
        return qis
    
    def _extract_zip_codes(self, text: str) -> list[QuasiIdentifier]:
        """Extract ZIP code mentions."""
        qis = []
        
        for pattern in self.ZIP_PATTERNS:
            for match in re.finditer(pattern, text):
                zip_code = match.group(1)
                # Validate it looks like a ZIP (not just any 5-digit number)
                # Check if preceded by ZIP-related context
                context_start = max(0, match.start() - 30)
                context = text[context_start:match.start()].lower()
                
                is_zip = any(word in context for word in 
                           ['zip', 'postal', 'address', 'located', 'lives', 'resides'])
                
                # Or if it's in a typical address format
                if not is_zip:
                    # Check for state abbreviation before it
                    state_pattern = r'[A-Z]{2}\s*$'
                    is_zip = bool(re.search(state_pattern, text[context_start:match.start()]))
                
                if is_zip or True:  # For now, capture all 5-digit numbers as potential ZIPs
                    qis.append(QuasiIdentifier(
                        qi_type=QIType.ZIP_CODE,
                        raw_value=match.group(0),
                        normalized_value=zip_code,
                        confidence=0.7 if not is_zip else 0.9,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return qis
    
    def _extract_diseases(self, text: str) -> list[QuasiIdentifier]:
        """Extract disease/condition mentions."""
        qis = []
        text_lower = text.lower()
        
        # Check for known diseases
        for disease in self.KNOWN_DISEASES:
            if disease in text_lower:
                # Find the position
                idx = text_lower.find(disease)
                qis.append(QuasiIdentifier(
                    qi_type=QIType.DISEASE,
                    raw_value=text[idx:idx+len(disease)],
                    normalized_value=disease,
                    confidence=0.95,
                    start_pos=idx,
                    end_pos=idx + len(disease)
                ))
        
        # Use patterns for other disease mentions
        for pattern in self.DISEASE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                disease = match.group(1).strip()
                # Filter out obvious non-diseases
                if len(disease) < 3 or disease.lower() in ['a', 'an', 'the', 'and']:
                    continue
                
                # Check if not already captured
                disease_lower = disease.lower()
                already_found = any(disease_lower in d for d in self.KNOWN_DISEASES)
                
                if not already_found:
                    qis.append(QuasiIdentifier(
                        qi_type=QIType.DISEASE,
                        raw_value=match.group(0),
                        normalized_value=disease.lower(),
                        confidence=0.7,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return qis
    
    def _extract_occupations(self, text: str) -> list[QuasiIdentifier]:
        """Extract occupation mentions."""
        qis = []
        
        # Common occupation words to look for directly
        common_occupations = [
            'doctor', 'physician', 'nurse', 'surgeon', 'lawyer', 'attorney',
            'teacher', 'professor', 'engineer', 'developer', 'programmer',
            'police officer', 'firefighter', 'pilot', 'chef', 'accountant',
            'dentist', 'pharmacist', 'therapist', 'scientist', 'researcher',
            'manager', 'executive', 'consultant', 'analyst'
        ]
        
        text_lower = text.lower()
        for occupation in common_occupations:
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
        
        # Use patterns for other mentions
        for pattern in self.OCCUPATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                occupation = match.group(1).strip()
                # Filter
                if len(occupation) < 3:
                    continue
                # Check not already found
                if occupation.lower() not in common_occupations:
                    qis.append(QuasiIdentifier(
                        qi_type=QIType.OCCUPATION,
                        raw_value=match.group(0),
                        normalized_value=occupation.lower(),
                        confidence=0.6,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return qis


# =============================================================================
# Q-Score Calculator
# =============================================================================

class QScoreCalculator:
    """
    Calculates the Q-Score based on detected quasi-identifiers
    and their population frequencies.
    """
    
    # Default US population
    REFERENCE_POPULATION = 331_900_000
    
    # k-anonymity threshold
    K_THRESHOLD = 5
    
    # Weights for combining Q-score components
    W1_RAREST = 0.3      # Weight for rarest single QI
    W2_COMBINATION = 0.5  # Weight for full combination
    W3_SUBSETS = 0.2      # Weight for identifying subsets
    
    def __init__(
        self,
        census_api_key: Optional[str] = None,
        bls_api_key: Optional[str] = None,
        orphanet_api_key: Optional[str] = None,
        k_threshold: int = 5
    ):
        """
        Initialize Q-Score calculator with data sources.
        
        Args:
            census_api_key: API key for Census Bureau
            bls_api_key: API key for BLS
            orphanet_api_key: API key for Orphanet
            k_threshold: Minimum k for k-anonymity (default: 5)
        """
        self.census_source = CensusDataSource(api_key=census_api_key)
        self.bls_source = BLSDataSource(api_key=bls_api_key)
        self.orphanet_source = OrphanetDataSource(api_key=orphanet_api_key)
        
        self.extractor = QIExtractor()
        self.k_threshold = k_threshold
    
    def calculate(self, text: str) -> QScoreResult:
        """
        Calculate the Q-Score for input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            QScoreResult with full analysis
        """
        # Step 1: Extract quasi-identifiers
        detected_qis = self.extractor.extract(text)
        
        if not detected_qis:
            return QScoreResult(
                q_score=0.0,
                q_rarest=0.0,
                q_combination=0.0,
                q_subsets=0.0,
                detected_qis=[],
                frequencies={},
                expected_k=self.REFERENCE_POPULATION,
                identifying_subsets=[],
                explanation="No quasi-identifiers detected."
            )
        
        # Step 2: Look up frequencies for each QI
        frequencies = {}
        for qi in detected_qis:
            freq = self._get_frequency(qi)
            if freq:
                frequencies[qi] = freq
        
        # Step 3: Calculate component scores
        q_rarest = self._calculate_rarest_score(frequencies)
        q_combination, expected_k = self._calculate_combination_score(frequencies)
        q_subsets, identifying_subsets = self._calculate_subset_score(frequencies)
        
        # Step 4: Combine scores
        q_score = (
            self.W1_RAREST * q_rarest +
            self.W2_COMBINATION * q_combination +
            self.W3_SUBSETS * q_subsets
        )
        
        # Ensure score is in [0, 1]
        q_score = max(0.0, min(1.0, q_score))
        
        # Generate explanation
        explanation = self._generate_explanation(
            detected_qis, frequencies, expected_k, identifying_subsets
        )
        
        return QScoreResult(
            q_score=q_score,
            q_rarest=q_rarest,
            q_combination=q_combination,
            q_subsets=q_subsets,
            detected_qis=detected_qis,
            frequencies=frequencies,
            expected_k=expected_k,
            identifying_subsets=identifying_subsets,
            explanation=explanation,
            weights={
                "w1_rarest": self.W1_RAREST,
                "w2_combination": self.W2_COMBINATION,
                "w3_subsets": self.W3_SUBSETS
            }
        )
    
    def _get_frequency(self, qi: QuasiIdentifier) -> Optional[QIFrequency]:
        """Get frequency for a quasi-identifier from appropriate source."""
        if qi.qi_type in [QIType.AGE, QIType.GENDER, QIType.LOCATION, QIType.ZIP_CODE]:
            return self.census_source.get_frequency(qi.qi_type, qi.normalized_value)
        elif qi.qi_type == QIType.DISEASE:
            return self.orphanet_source.get_frequency(qi.qi_type, qi.normalized_value)
        elif qi.qi_type == QIType.OCCUPATION:
            return self.bls_source.get_frequency(qi.qi_type, qi.normalized_value)
        return None
    
    def _calculate_rarest_score(self, frequencies: dict) -> float:
        """
        Calculate score based on the rarest single quasi-identifier.
        
        A very rare single attribute (like a rare disease) is already
        a significant re-identification risk.
        """
        if not frequencies:
            return 0.0
        
        # Find minimum probability
        min_prob = min(f.probability for f in frequencies.values())
        
        # Convert to risk score
        # If probability is very low (e.g., 1 in 100,000), risk is high
        expected_k_single = self.REFERENCE_POPULATION * min_prob
        
        if expected_k_single >= self.k_threshold:
            return 0.0
        else:
            # Linear scaling: 0 at k_threshold, 1 at k=1
            return 1.0 - (expected_k_single / self.k_threshold)
    
    def _calculate_combination_score(self, frequencies: dict) -> tuple[float, float]:
        """
        Calculate score based on the combination of all quasi-identifiers.
        
        Uses independence assumption: π(v1, v2, ..., vm) ≈ Π π(vi)
        
        Returns:
            Tuple of (combination_score, expected_k)
        """
        if not frequencies:
            return 0.0, self.REFERENCE_POPULATION
        
        # Calculate joint probability under independence
        joint_prob = 1.0
        for freq in frequencies.values():
            joint_prob *= freq.probability
        
        # Expected equivalence class size
        expected_k = self.REFERENCE_POPULATION * joint_prob
        
        # Convert to risk score using equation (4) from the paper
        if expected_k >= self.k_threshold:
            combination_score = 0.0
        else:
            combination_score = 1.0 - (expected_k / self.k_threshold)
        
        return combination_score, expected_k
    
    def _calculate_subset_score(self, frequencies: dict) -> tuple[float, list]:
        """
        Check if any subset of QIs is already identifying.
        
        Sometimes a pair of attributes is uniquely identifying even
        without the full combination.
        
        Returns:
            Tuple of (subset_score, list of identifying subsets)
        """
        if len(frequencies) < 2:
            return 0.0, []
        
        identifying_subsets = []
        max_subset_risk = 0.0
        
        qis = list(frequencies.keys())
        
        # Check all subsets of size 2 to n-1
        for size in range(2, len(qis)):
            for subset in combinations(qis, size):
                # Calculate expected k for this subset
                joint_prob = 1.0
                for qi in subset:
                    joint_prob *= frequencies[qi].probability
                
                expected_k = self.REFERENCE_POPULATION * joint_prob
                
                if expected_k < self.k_threshold:
                    subset_risk = 1.0 - (expected_k / self.k_threshold)
                    max_subset_risk = max(max_subset_risk, subset_risk)
                    
                    identifying_subsets.append({
                        'qis': [qi.normalized_value for qi in subset],
                        'types': [qi.qi_type.value for qi in subset],
                        'expected_k': expected_k,
                        'risk': subset_risk
                    })
        
        return max_subset_risk, identifying_subsets
    
    def _generate_explanation(
        self,
        detected_qis: list,
        frequencies: dict,
        expected_k: float,
        identifying_subsets: list
    ) -> str:
        """Generate human-readable explanation of the Q-Score."""
        lines = []
        
        # Summarize detected QIs
        lines.append(f"Detected {len(detected_qis)} quasi-identifier(s):")
        for qi in detected_qis:
            freq = frequencies.get(qi)
            if freq:
                lines.append(
                    f"  - {qi.qi_type.value}: '{qi.normalized_value}' "
                    f"(π ≈ {freq.probability:.2e}, ~{freq.population_count:,} people)"
                )
            else:
                lines.append(f"  - {qi.qi_type.value}: '{qi.normalized_value}' (frequency unknown)")
        
        lines.append("")
        lines.append(f"Expected equivalence class size (E[k]): {expected_k:.2f}")
        
        if expected_k < 1:
            lines.append("⚠️  WARNING: E[k] < 1 means this combination is likely UNIQUELY IDENTIFYING")
        elif expected_k < self.k_threshold:
            lines.append(f"⚠️  WARNING: E[k] < {self.k_threshold} indicates HIGH re-identification risk")
        else:
            lines.append(f"✓ E[k] ≥ {self.k_threshold} suggests acceptable k-anonymity")
        
        if identifying_subsets:
            lines.append("")
            lines.append(f"Found {len(identifying_subsets)} identifying subset(s):")
            for subset in identifying_subsets[:3]:  # Show top 3
                lines.append(
                    f"  - {subset['qis']} → E[k] = {subset['expected_k']:.2f}"
                )
        
        return "\n".join(lines)
    
    def get_detailed_report(self, text: str) -> dict:
        """
        Generate a detailed JSON-serializable report.
        Useful for audit trails per HIPAA requirements.
        """
        result = self.calculate(text)
        
        return {
            "input_length": len(text),
            "scores": {
                "q_score": result.q_score,
                "q_rarest": result.q_rarest,
                "q_combination": result.q_combination,
                "q_subsets": result.q_subsets
            },
            "risk_level": self._get_risk_level(result.q_score),
            "expected_k": result.expected_k,
            "k_threshold": self.k_threshold,
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
                        "source": result.frequencies[qi].source,
                        "is_estimated": result.frequencies[qi].is_estimated
                    } if qi in result.frequencies else None
                }
                for qi in result.detected_qis
            ],
            "identifying_subsets": result.identifying_subsets,
            "weights_used": result.weights,
            "explanation": result.explanation
        }
    
    def _get_risk_level(self, q_score: float) -> str:
        """Convert Q-score to risk level."""
        if q_score >= 0.7:
            return "HIGH"
        elif q_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"


# =============================================================================
# Main Interface
# =============================================================================

class QScoreAnalyzer:
    """
    Main interface for Q-Score analysis.
    Provides a simple API for the sensitivity classification pipeline.
    """
    
    def __init__(
        self,
        census_api_key: Optional[str] = None,
        bls_api_key: Optional[str] = None,
        orphanet_api_key: Optional[str] = None,
        k_threshold: int = 5
    ):
        """
        Initialize the Q-Score analyzer.
        
        Args:
            census_api_key: Census Bureau API key
            bls_api_key: Bureau of Labor Statistics API key
            orphanet_api_key: Orphanet API key
            k_threshold: Minimum k for k-anonymity
        """
        self.calculator = QScoreCalculator(
            census_api_key=census_api_key,
            bls_api_key=bls_api_key,
            orphanet_api_key=orphanet_api_key,
            k_threshold=k_threshold
        )
    
    def analyze(self, text: str) -> QScoreResult:
        """
        Analyze text for quasi-identifier risk.
        
        Args:
            text: Input text to analyze
            
        Returns:
            QScoreResult with complete analysis
        """
        return self.calculator.calculate(text)
    
    def get_score(self, text: str) -> float:
        """
        Get just the Q-Score value.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Q-Score between 0 and 1
        """
        return self.calculator.calculate(text).q_score
    
    def is_identifying(self, text: str) -> bool:
        """
        Check if the text contains an identifying combination.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if E[k] < 1 (uniquely identifying)
        """
        result = self.calculator.calculate(text)
        return result.expected_k < 1
    
    def get_audit_report(self, text: str) -> dict:
        """
        Get a detailed audit report (for HIPAA compliance).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with complete analysis details
        """
        return self.calculator.get_detailed_report(text)
    def analyze_to_json(
        self, 
        text: str, 
        output_path: str = None,
        pretty_print: bool = True
    ) -> str:
        """
        Analyze text and save results to a JSON file.
        
        Args:
            text: Input text to analyze
            output_path: Path for output JSON file. If None, auto-generates filename.
            pretty_print: If True, format JSON with indentation
            
        Returns:
            Path to the created JSON file
        """
        # Get the detailed report
        report = self.calculator.get_detailed_report(text)
        
        # Add metadata
        report["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "input_text": text,
            "analyzer_version": "1.0.0"
        }
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"qscore_report_{timestamp}.json"
        
        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)
            else:
                json.dump(report, f, default=str, ensure_ascii=False)
        
        return output_path
    
    def analyze_batch_to_json(
        self,
        texts: list[str],
        output_path: str = "qscore_batch_report.json",
        pretty_print: bool = True
    ) -> str:
        """
        Analyze multiple texts and save all results to a single JSON file.
        
        Args:
            texts: List of input texts to analyze
            output_path: Path for output JSON file
            pretty_print: If True, format JSON with indentation
            
        Returns:
            Path to the created JSON file
        """
        batch_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_inputs": len(texts),
                "analyzer_version": "1.0.0"
            },
            "summary": {
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "uniquely_identifying_count": 0
            },
            "results": []
        }
        
        for i, text in enumerate(texts):
            report = self.calculator.get_detailed_report(text)
            report["input_index"] = i
            report["input_text"] = text
            
            # Update summary
            if report["expected_k"] < 1:
                batch_report["summary"]["uniquely_identifying_count"] += 1
            
            risk_level = report["risk_level"]
            if risk_level == "HIGH":
                batch_report["summary"]["high_risk_count"] += 1
            elif risk_level == "MEDIUM":
                batch_report["summary"]["medium_risk_count"] += 1
            else:
                batch_report["summary"]["low_risk_count"] += 1
            
            batch_report["results"].append(report)
        
        # Write JSON file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(batch_report, f, indent=2, default=str, ensure_ascii=False)
            else:
                json.dump(batch_report, f, default=str, ensure_ascii=False)
        
        return output_path


def main():
    """Demonstrate Q-Score analysis with example texts."""
    
    # Initialize analyzer
    analyzer = QScoreAnalyzer(
        census_api_key="43b79ffec467dfaf3c219355e374a75e193f927a",
        bls_api_key="5172554501df486283660c43278ca891",
        orphanet_api_key="your_orphanet_api_key_here",  # if you have one
        k_threshold=5
    )
    
    examples = load_example_inputs()
    test_cases = [example["text"] for example in examples]
    
    output_path = analyzer.analyze_batch_to_json(
        test_cases,
        output_path="qscore_demo_report.json",
        pretty_print=True,
    )

    print(f"JSON report written to: {output_path}")
    # Also write a unified detections JSON in outputs/
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    unified = []

    for ex in examples:
        report = analyzer.get_audit_report(ex['text'])
        # detected_quasi_identifiers is already a list of dicts
        detections = []
        for d in report.get('detected_quasi_identifiers', []):
            detections.append({
                'type': d.get('type'),
                'raw_value': d.get('raw_value'),
                'normalized_value': d.get('normalized_value'),
                'confidence': d.get('confidence'),
                'start': d.get('position', {}).get('start') if d.get('position') else None,
                'end': d.get('position', {}).get('end') if d.get('position') else None,
            })

        unified.append({
            'name': ex.get('name'),
            'text': ex.get('text'),
            'expected_critical': ex.get('expected_critical', False),
            'detections': detections,
            'expected_k': report.get('expected_k')
        })

    out_file = out_dir / 'q_score_unified.json'
    out_file.write_text(json.dumps(unified, indent=2, ensure_ascii=False))
    print(f"Unified Q-Score JSON written to: {out_file}")


if __name__ == "__main__":
    main()