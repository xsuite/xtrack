from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os


class FieldMapParser(ABC):
    """Base class for field map file parsers."""
    
    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Parse field map file into standardized DataFrame.
        
        Returns:
            DataFrame with MultiIndex (X, Y, Z) and columns (Bx, By, Bs)
        """
        pass
    
    @classmethod
    def detect_format(cls, file_path: str) -> bool:
        """Detect if this parser can handle the given file."""
        return False


class StandardFieldMapParser(FieldMapParser):
    """Parser for standard 6-column format: X Y Z Bx By Bs"""
    
    def parse(self, file_path: str, **kwargs) -> pd.DataFrame:
        df = pd.read_csv(
            file_path, sep=r"\s+", header=None, 
            names=["X", "Y", "Z", "Bx", "By", "Bs"]
        )
        df.set_index(["X", "Y", "Z"], inplace=True)
        return df
    
    @classmethod
    def detect_format(cls, file_path: str) -> bool:
        # Check first line has 6 columns
        try:
            # Try to open the file (handles both absolute and relative paths)
            # If relative, it will be resolved relative to current working directory
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                parts = first_line.split()
                return len(parts) == 6
        except (FileNotFoundError, OSError, IOError):
            # If file doesn't exist or can't be read, can't detect format
            # This is expected for relative paths when CWD differs from script location
            return False


class UE36FieldMapParser(FieldMapParser):
    """Parser for UE36 format: Z, then Bx/By/Bs for each Y position"""
    
    def __init__(self, y_positions=None, field_order='Bx,By,Bs', dx=0.001, dy=0.001):
        """
        Args:
            y_positions: Array of Y positions (if None, will infer from data)
            field_order: Order of fields in columns (e.g., 'Bx,By,Bs' or 'By,Bx,Bs')
            dx: X spacing (if needed for coordinate reconstruction)
            dy: Y spacing (if needed for coordinate reconstruction)
        """
        self.y_positions = y_positions
        self.field_order = field_order.split(',')
        self.dx = dx
        self.dy = dy
    
    def parse(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Parse UE36 format file.
        
        Format: First column is Z, followed by field values for each Y position.
        Each Y position has 3 values (Bx, By, Bs) in the order specified by field_order.
        """
        # Get dx and dy from kwargs or use instance defaults
        dx = kwargs.get('dx', self.dx)
        dy = kwargs.get('dy', self.dy)
        
        # Read the file - tab-separated
        data = np.loadtxt(file_path, delimiter='\t')
        
        # First column is Z
        z_values = data[:, 0]
        
        # Remaining columns are field values
        field_data = data[:, 1:]
        
        # Determine number of Y positions
        # Each Y position has 3 fields (Bx, By, Bs)
        n_y_positions = field_data.shape[1] // 3
        
        if field_data.shape[1] % 3 != 0:
            raise ValueError(f"Number of field columns ({field_data.shape[1]}) is not divisible by 3")
        
        # Infer Y positions if not provided
        if self.y_positions is None:
            # Assume symmetric around 0, with spacing dy
            # We need to infer the number of Y positions and their values
            # For now, create symmetric positions around 0
            y_min = -(n_y_positions - 1) / 2 * dy
            y_positions = np.linspace(y_min, -y_min, n_y_positions)
        else:
            y_positions = np.asarray(self.y_positions)
            if len(y_positions) != n_y_positions:
                raise ValueError(f"Number of provided Y positions ({len(y_positions)}) "
                               f"does not match data ({n_y_positions})")
        
        # Extract field values for each Y position
        # Field order: Bx, By, Bs (or as specified)
        field_map = {field: i for i, field in enumerate(self.field_order)}
        
        # Build list of rows for DataFrame
        rows = []
        for z_idx, z_val in enumerate(z_values):
            for y_idx, y_val in enumerate(y_positions):
                # Extract field values for this (Z, Y) combination
                col_start = y_idx * 3
                field_values = field_data[z_idx, col_start:col_start + 3]
                
                # Map to Bx, By, Bs according to field_order
                bx = field_values[field_map['Bx']]
                by = field_values[field_map['By']]
                bs = field_values[field_map['Bs']]
                
                # For UE36 format, X is typically 0 (on-axis)
                # If we need to support multiple X positions, we'd need more info
                x_val = 0.0
                
                rows.append({
                    'X': x_val,
                    'Y': y_val,
                    'Z': z_val,
                    'Bx': bx,
                    'By': by,
                    'Bs': bs
                })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        df.set_index(["X", "Y", "Z"], inplace=True)
        return df
    
    @classmethod
    def detect_format(cls, file_path: str) -> bool:
        """Detect UE36 format by checking if first column looks like Z and has many columns."""
        try:
            # Try to open the file (handles both absolute and relative paths)
            # If relative, it will be resolved relative to current working directory
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                parts = first_line.split('\t')  # UE36 format is tab-separated
                if len(parts) < 10:  # Should have many columns (Z + many field values)
                    return False
                # Check if first column could be a Z value (numeric)
                try:
                    float(parts[0])
                    # Check if we have many columns (suggesting UE36 format)
                    # Standard format has 6 columns, UE36 has many more
                    return len(parts) > 20
                except ValueError:
                    return False
        except (FileNotFoundError, OSError, IOError):
            # If file doesn't exist or can't be read, can't detect format
            # This is expected for relative paths when CWD differs from script location
            return False
        except Exception:
            return False


def get_parser_for_file(file_path: str) -> FieldMapParser:
    """Auto-detect and return appropriate parser."""
    parsers = [SolenoidFieldMapParser, StandardFieldMapParser, UE36FieldMapParser]
    for parser_class in parsers:
        if parser_class.detect_format(file_path):
            return parser_class()
    raise ValueError(f"Could not detect format for {file_path}")
