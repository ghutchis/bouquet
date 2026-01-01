"""Unit tests for bouquet.io module."""

import json
import logging
from pathlib import Path

import pytest
from ase import Atoms

from bouquet.io import (
    setup_logging,
    create_output_directory,
    save_run_parameters,
    save_structure,
    atoms_to_xyz_string,
    initialize_structure_log,
    create_structure_logger,
)


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_creates_log_file(self, temp_dir):
        """Test that setup_logging creates a runtime.log file."""
        logger = setup_logging(temp_dir)
        log_file = temp_dir / "runtime.log"
        assert log_file.exists()

    def test_returns_logger(self, temp_dir):
        """Test that setup_logging returns a logger instance."""
        logger = setup_logging(temp_dir)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "bouquet"

    def test_custom_logger_name(self, temp_dir):
        """Test setup_logging with custom logger name."""
        logger = setup_logging(temp_dir, logger_name="custom_test")
        assert logger.name == "custom_test"

    def test_logger_writes_to_file(self, temp_dir):
        """Test that logger writes messages to file."""
        import logging as log_module

        # Clear any existing handlers to avoid conflicts
        log_module.root.handlers.clear()

        logger = setup_logging(temp_dir, logger_name="test_write_unique")
        logger.info("Test message")

        # Force handlers to flush and close
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

        log_content = (temp_dir / "runtime.log").read_text()
        assert "Test message" in log_content


class TestCreateOutputDirectory:
    """Tests for the create_output_directory function."""

    def test_creates_directory(self, temp_dir, monkeypatch):
        """Test that output directory is created."""
        monkeypatch.chdir(temp_dir)

        args_dict = {"test": "value"}
        out_dir = create_output_directory("test_mol", 42, "gfn2", args_dict)

        assert out_dir.exists()
        assert out_dir.is_dir()

    def test_directory_name_format(self, temp_dir, monkeypatch):
        """Test that directory name follows expected format."""
        monkeypatch.chdir(temp_dir)

        args_dict = {"test": "value"}
        out_dir = create_output_directory("butane", 123, "gfn2", args_dict)

        # Should contain name-seed-method-hash
        assert "butane" in out_dir.name
        assert "123" in out_dir.name
        assert "gfn2" in out_dir.name

    def test_hash_changes_with_args(self, temp_dir, monkeypatch):
        """Test that different args produce different hashes."""
        monkeypatch.chdir(temp_dir)

        dir1 = create_output_directory("mol", 1, "gfn2", {"a": 1})
        dir2 = create_output_directory("mol", 1, "gfn2", {"a": 2})

        # Last 6 characters (hash) should differ
        assert dir1.name != dir2.name

    def test_creates_parent_directories(self, temp_dir, monkeypatch):
        """Test that parent 'solutions' directory is created."""
        monkeypatch.chdir(temp_dir)

        args_dict = {"test": "value"}
        out_dir = create_output_directory("test", 1, "gfn2", args_dict)

        assert (temp_dir / "solutions").exists()


class TestSaveRunParameters:
    """Tests for the save_run_parameters function."""

    def test_saves_json_file(self, temp_dir):
        """Test that parameters are saved as JSON."""
        args_dict = {"smiles": "CCCC", "method": "gfn2", "steps": 32}
        save_run_parameters(temp_dir, args_dict)

        params_file = temp_dir / "run_params.json"
        assert params_file.exists()

    def test_json_content_matches(self, temp_dir):
        """Test that saved JSON matches input dictionary."""
        args_dict = {"smiles": "CCCC", "method": "gfn2", "steps": 32}
        save_run_parameters(temp_dir, args_dict)

        with (temp_dir / "run_params.json").open() as f:
            loaded = json.load(f)

        assert loaded == args_dict


class TestSaveStructure:
    """Tests for the save_structure function."""

    def test_saves_xyz_file(self, temp_dir, simple_atoms):
        """Test that structure is saved as XYZ file."""
        save_structure(temp_dir, simple_atoms, "test.xyz")

        xyz_file = temp_dir / "test.xyz"
        assert xyz_file.exists()

    def test_xyz_content_format(self, temp_dir, simple_atoms):
        """Test that XYZ file has correct format."""
        save_structure(temp_dir, simple_atoms, "test.xyz")

        content = (temp_dir / "test.xyz").read_text()
        lines = content.strip().split("\n")

        # First line should be atom count
        assert lines[0].strip() == "3"
        # Third line onwards should have atom positions
        assert lines[2].startswith("O")

    def test_xyz_with_comment(self, temp_dir, simple_atoms):
        """Test that comment is included in XYZ file."""
        save_structure(temp_dir, simple_atoms, "test.xyz", comment="Test comment")

        content = (temp_dir / "test.xyz").read_text()
        lines = content.split("\n")

        # Second line is the comment
        assert "Test comment" in lines[1]


class TestAtomsToXyzString:
    """Tests for the atoms_to_xyz_string function."""

    def test_returns_string(self, simple_atoms):
        """Test that function returns a string."""
        result = atoms_to_xyz_string(simple_atoms)
        assert isinstance(result, str)

    def test_xyz_format(self, simple_atoms):
        """Test that string is in XYZ format."""
        result = atoms_to_xyz_string(simple_atoms)
        lines = result.strip().split("\n")

        # First line is atom count
        assert lines[0].strip() == "3"
        # Should have atoms
        assert "O" in result
        assert "H" in result

    def test_atom_count_matches(self, ethane_atoms):
        """Test that atom count in XYZ matches atoms object."""
        result = atoms_to_xyz_string(ethane_atoms)
        lines = result.strip().split("\n")

        assert int(lines[0].strip()) == len(ethane_atoms)


class TestInitializeStructureLog:
    """Tests for the initialize_structure_log function."""

    def test_creates_csv_file(self, temp_dir):
        """Test that CSV log file is created."""
        log_path, ens_path = initialize_structure_log(temp_dir)

        assert log_path.exists()
        assert log_path.name == "structures.csv"

    def test_returns_paths(self, temp_dir):
        """Test that function returns correct paths."""
        log_path, ens_path = initialize_structure_log(temp_dir)

        assert log_path == temp_dir / "structures.csv"
        assert ens_path == temp_dir / "ensemble.xyz"

    def test_csv_has_headers(self, temp_dir):
        """Test that CSV file has correct headers."""
        log_path, _ = initialize_structure_log(temp_dir)

        content = log_path.read_text()
        assert "time" in content
        assert "xyz" in content
        assert "energy" in content
        assert "ediff" in content


class TestCreateStructureLogger:
    """Tests for the create_structure_logger function."""

    def test_returns_callable(self, temp_dir):
        """Test that function returns a callable."""
        log_path, ens_path = initialize_structure_log(temp_dir)
        add_entry = create_structure_logger(log_path, ens_path, -100.0)

        assert callable(add_entry)

    def test_logs_entry_to_csv(self, temp_dir, simple_atoms):
        """Test that entry is logged to CSV file."""
        import numpy as np

        log_path, ens_path = initialize_structure_log(temp_dir)
        add_entry = create_structure_logger(log_path, ens_path, -100.0)

        coords = np.array([180.0, 60.0])
        add_entry(coords, simple_atoms, -99.5)

        content = log_path.read_text()
        # Should have the energy value
        assert "-99.5" in content
        # Should have energy difference (0.5)
        assert "0.5" in content

    def test_logs_entry_to_ensemble(self, temp_dir, simple_atoms):
        """Test that entry is added to ensemble XYZ file."""
        import numpy as np

        log_path, ens_path = initialize_structure_log(temp_dir)
        add_entry = create_structure_logger(log_path, ens_path, -100.0)

        coords = np.array([180.0])
        add_entry(coords, simple_atoms, -99.0)

        content = ens_path.read_text()
        # Should have atom count
        assert "3" in content
        # Should have energy in comment
        assert "-99.0" in content

    def test_multiple_entries(self, temp_dir, simple_atoms):
        """Test logging multiple entries."""
        import csv
        import numpy as np

        log_path, ens_path = initialize_structure_log(temp_dir)
        add_entry = create_structure_logger(log_path, ens_path, -100.0)

        for i in range(3):
            coords = np.array([180.0 + i * 10])
            add_entry(coords, simple_atoms, -99.0 + i)

        # CSV should have header + 3 entries (use csv reader to handle embedded newlines)
        with log_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3  # 3 entries (header is not counted by DictReader)

        # Ensemble should have 3 structures
        ens_content = ens_path.read_text()
        # Count number of structures - first one starts at beginning, rest have \n3\n
        # The pattern is: "3\n" at start + "\n3\n" for subsequent structures
        structure_count = 1 + ens_content.count("\n3\n")  # first structure + rest
        assert structure_count == 3  # 3 structures with 3 atoms each
