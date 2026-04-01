from .financial import load_financial, METADATA as FINANCIAL_META
from .clinical  import load_clinical,  METADATA as CLINICAL_META

PROBLEM_SPACES = [
    {"key": "financial", "loader": load_financial, **FINANCIAL_META},
    {"key": "clinical",  "loader": load_clinical,  **CLINICAL_META},
]
