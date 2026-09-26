from dataclasses import dataclass

@dataclass
class RNAConfig:
    """Global configuration for the RNAcompete Data Pipeline."""
    
    # Data Path
    DATA_PATH: str = "data/norm_data.txt"

    # Metadata is an Excel file
    METADATA_PATH: str = "data/metadata.xlsx"
    METADATA_SHEET: str = "Master List--Plasmid Info"
    
    # Save Path
    SAVE_DIR: str = "data"
    
    # Sequence Parameters
    SEQ_MAX_LEN: int = 41
    ALPHABET: str = "ACGUN"
    
    # Preprocessing
    CLIP_PERCENTILE: float = 99.95
    EPSILON: float = 1e-6  # For numerical stability
    
    # Split Identifiers
    TRAIN_SPLIT_ID: str = "SetA"
    TEST_SPLIT_ID: str = "SetB"
    
    VAL_SPLIT_PCT: float = 0.2
    SEED: int = 42 # NOTE: Change this only if you want to test reproducibility