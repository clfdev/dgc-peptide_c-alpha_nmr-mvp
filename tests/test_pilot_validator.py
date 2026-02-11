# tests/test_pilot_validator.py

import pytest
from src.validation.pilot_validator import (
    PilotValidator, 
    ValidationStatus,
    ValidationResult
)


class TestPilotValidator:
    """Testes unitários para o validador."""
    
    @pytest.fixture
    def validator(self):
        return PilotValidator()
    
    def test_pdb_availability_valid(self, validator):
        """Testa check de PDB válido."""
        result = validator._check_pdb_availability('1UBQ')
        assert result.status == ValidationStatus.PASS
    
    def test_pdb_availability_invalid(self, validator):
        """Testa check de PDB inválido."""
        result = validator._check_pdb_availability('XXXX')
        assert result.status == ValidationStatus.FAIL
    
    def test_bmrb_availability_valid(self, validator):
        """Testa check de BMRB válido."""
        result = validator._check_bmrb_availability('6457')
        assert result.status == ValidationStatus.PASS
    
    def test_full_validation_ubiquitin(self, validator):
        """Testa validação completa da ubiquitina."""
        report = validator.validate_pilot_entry('1UBQ', '6457')
        
        # Deve passar em checks básicos
        assert report.checks['1_pdb_exists'].status == ValidationStatus.PASS
        assert report.checks['2_bmrb_exists'].status == ValidationStatus.PASS
        
        # Ubiquitina tem 76 resíduos (acima do limite)
        # Mas vamos aceitar com WARNING
        print(report)