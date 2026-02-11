# scripts/validate_new_candidates.py

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from validation.pilot_validator import validate_pilot_list

# Novos candidatos com chemical shifts CA confirmados
NEW_CANDIDATES = [
    ('1LE1', '34305'),  # Trpzip-2
    ('4CZ3', '19911'),  # Candidato 2
    ('4CZ4', '19929'),  # Candidato 3
    ('5W9F', '30312'),  # Candidato 4
]

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VALIDAÇÃO DE NOVOS CANDIDATOS COM CA SHIFTS CONFIRMADOS")
    print("="*70 + "\n")
    
    validate_pilot_list(NEW_CANDIDATES)