from pathlib import Path
import json
import math

def to_scientific_notation(number:float)->str:
    """
    Converts number to scientific notation with one digit.
    For instance, 5000 becomes '5e3' and 123.45 becomes '1.2e2'
    """
    exponent = int(round(math.log10(number)))
    mantissa = round(number / (10 ** exponent), 1)
    
    # Format as string
    return f"{mantissa}e{exponent}"

def get_project_root() -> Path:
    """
    Returns always project root directory.
    """
    return Path(__file__).parent.parent

