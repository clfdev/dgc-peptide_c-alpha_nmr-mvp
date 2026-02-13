#!/usr/bin/env python3
# scripts/pilot_list.py

"""
Pilot dataset definition for Phase 0 MVP.

This module defines the curated list of validated PDB-BMRB pairs
used for training and evaluation of the geometric chemical shift predictor.

Usage:
    from scripts.pilot_list import PILOT_STRUCTURES, get_pilot_list
    
    structures = get_pilot_list()
    for entry in structures:
        print(f"{entry['pdb_id']} <-> BMRB {entry['bmrb_id']}")
"""

# Validated PDB-BMRB pairs (all passed pilot_validator.py checks)
PILOT_STRUCTURES = [
    {
        'pdb_id': '4CZ3',
        'bmrb_id': '19911',
        'description': 'Peptide structure',
        'expected_length': None,  # To be determined from PDB
        'notes': 'Approved'
    },
    {
        'pdb_id': '1LE1',
        'bmrb_id': '34305',
        'description': 'Trpzip-2 beta-hairpin',
        'expected_length': 12,
        'notes': 'Approved - BMRB ID corrected from 5387'
    },
    {
        'pdb_id': '4CZ4',
        'bmrb_id': '19929',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved'
    },
    {
        'pdb_id': '5W9F',
        'bmrb_id': '30312',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved'
    },
    {
        'pdb_id': '1AQ5',
        'bmrb_id': '4055',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition'
    },
    {
        'pdb_id': '6R9Z',
        'bmrb_id': '34391',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition'
    },
    {
        'pdb_id': '2LHY',
        'bmrb_id': '17871',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition'
    },
    {
        'pdb_id': '2JVD',
        'bmrb_id': '15476',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition'
    },
    {
        'pdb_id': '1S6W',
        'bmrb_id': '6085',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition'
    },
    {
        'pdb_id': '6DST',
        'bmrb_id': '30481',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition'
    },
    {
        'pdb_id': '1BHI',
        'bmrb_id': '4216',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '1CR8',
        'bmrb_id': '4475',
        'description': 'Peptide structure', 
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '2AP7',
        'bmrb_id': '6774',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '2B88',
        'bmrb_id': '6804',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '4BD3',
        'bmrb_id': '18764',
        'description': 'Peptide structure', 
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '3ZKT',
        'bmrb_id': '18972',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '5FZW',
        'bmrb_id': '26003',
        'description': 'Peptide structure', 
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '5IJ4',
        'bmrb_id': '30030',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '5LO4',
        'bmrb_id': '34033',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '6BVX',
        'bmrb_id': '30382',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    },
    {
        'pdb_id': '6F3Y',
        'bmrb_id': '34209',
        'description': 'Peptide structure',
        'expected_length': None,
        'notes': 'Approved - New addition2'
    }
]


# Directory paths for data storage
DATA_PATHS = {
    'raw_pdb': 'data/raw/pdb',
    'raw_bmrb': 'data/raw/bmrb',
    'processed': 'data/processed'
}


# URL templates for manual download reference
URL_TEMPLATES = {
    'pdb': 'https://files.rcsb.org/download/{pdb_id}.pdb',
    'bmrb': 'https://bmrb.io/ftp/pub/bmrb/entry_lists/nmr-star3.1/bmr{bmrb_id}.str'
}


def get_pilot_list():
    """
    Get the list of validated pilot structures.
    
    Returns:
        list: List of dictionaries containing PDB-BMRB pair information
    """
    return PILOT_STRUCTURES


def get_data_paths():
    """
    Get the standard data directory paths.
    
    Returns:
        dict: Dictionary mapping data types to directory paths
    """
    return DATA_PATHS


def get_url_templates():
    """
    Get URL templates for manual downloads.
    
    Returns:
        dict: Dictionary with PDB and BMRB URL templates
    """
    return URL_TEMPLATES


def print_pilot_summary():
    """Print a summary of the pilot dataset."""
    print("="*70)
    print("PILOT DATASET SUMMARY")
    print("="*70)
    print(f"\nTotal structures: {len(PILOT_STRUCTURES)}\n")
    
    for i, entry in enumerate(PILOT_STRUCTURES, 1):
        pdb_id = entry['pdb_id']
        bmrb_id = entry['bmrb_id']
        desc = entry['description']
        
        print(f"{i:2d}. {pdb_id} <-> BMRB {bmrb_id}")
        if entry['expected_length']:
            print(f"    Length: {entry['expected_length']} residues")
        if entry['notes']:
            print(f"    Notes: {entry['notes']}")
        print()
    
    print("="*70)
    print("\nDATA DIRECTORY STRUCTURE:")
    for key, path in DATA_PATHS.items():
        print(f"  {key}: {path}/")
    
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("="*70)
    print("\nPDB files:")
    print(f"  URL: {URL_TEMPLATES['pdb']}")
    print(f"  Example: {URL_TEMPLATES['pdb'].format(pdb_id='1LE1')}")
    
    print("\nBMRB files:")
    print(f"  URL: {URL_TEMPLATES['bmrb']}")
    print(f"  Example: {URL_TEMPLATES['bmrb'].format(bmrb_id='34305')}")
    
    print("\n" + "="*70)


def generate_download_script(output_file='download_commands.sh'):
    """
    Generate a bash script with wget commands for manual download.
    
    Args:
        output_file: Path to output shell script
    """
    with open(output_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Automated download script for pilot dataset\n")
        f.write("# Generated by pilot_list.py\n\n")
        
        f.write("# Create directories\n")
        f.write("mkdir -p data/raw/pdb\n")
        f.write("mkdir -p data/raw/bmrb\n\n")
        
        f.write("# Download PDB files\n")
        f.write("echo 'Downloading PDB files...'\n")
        for entry in PILOT_STRUCTURES:
            pdb_id = entry['pdb_id']
            url = URL_TEMPLATES['pdb'].format(pdb_id=pdb_id)
            f.write(f"wget -q -O data/raw/pdb/{pdb_id}.pdb {url}\n")
        
        f.write("\n# Download BMRB files\n")
        f.write("echo 'Downloading BMRB files...'\n")
        for entry in PILOT_STRUCTURES:
            bmrb_id = entry['bmrb_id']
            url = URL_TEMPLATES['bmrb'].format(bmrb_id=bmrb_id)
            f.write(f"wget -q -O data/raw/bmrb/bmr{bmrb_id}.str {url}\n")
        
        f.write("\necho 'Download complete!'\n")
    
    print(f"Download script saved to: {output_file}")
    print(f"Make executable with: chmod +x {output_file}")
    print(f"Run with: ./{output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pilot dataset definition and download helper'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary of pilot dataset'
    )
    parser.add_argument(
        '--generate-script',
        action='store_true',
        help='Generate bash download script'
    )
    
    args = parser.parse_args()
    
    if args.summary or not any(vars(args).values()):
        print_pilot_summary()
    
    if args.generate_script:
        print("\n")
        generate_download_script()