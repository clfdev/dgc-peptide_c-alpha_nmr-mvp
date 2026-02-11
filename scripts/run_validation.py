# scripts/run_validation.py

"""
Valida a lista piloto de pares PDB-BMRB.

Uso:
    python scripts/run_validation.py
    python scripts/run_validation.py --output results/validation.txt
"""

import sys
import argparse
from pathlib import Path


# Adicionar src/ ao path (alternativa ao pip install -e .)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from validation.pilot_validator import validate_pilot_list


def main():
    parser = argparse.ArgumentParser(
        description='Validar lista piloto PDB-BMRB'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/validation_report.txt',
        help='Arquivo de saída do relatório'
    )
    args = parser.parse_args()
    
    # Lista piloto
    pilot_entries = [
        ('1VII', '5713'),  # Villin headpiece
        ('1LE1', '5387'),  # Trpzip-2
        ('1E0L', '4737'),  # WW domain
        ('1UBQ', '6457'),  # Ubiquitin
        ('2MAG', '6031'),  # Magainin-2
    ]
    
    # Criar diretório de resultados se não existir
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Redirecionar output para arquivo
    with open(args.output, 'w') as f:
        sys.stdout = f
        validate_pilot_list(pilot_entries)
        sys.stdout = sys.__stdout__  # Restaurar stdout
    
    print(f"✓ Relatório salvo em: {args.output}")


if __name__ == "__main__":
    main()
