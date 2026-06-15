
from dicom_processor import complete_standardization_workflow
from configs import BATCH_DIR, LABELS_CSV, ORIGINAL_DIR, CACHE_PATH, MAIN_PATH

# BATCH_DIR = MAIN_PATH + 'batches'
# ORIGINAL_DIR = MAIN_PATH + 'test'
# CACHE_PATH = MAIN_PATH + "test_cache.pkl"
original_volume_paths = complete_standardization_workflow(
        batch_dir=BATCH_DIR,
        labels_csv=LABELS_CSV,
        output_base_dir=ORIGINAL_DIR,
        pkl_path=CACHE_PATH,
        overwrite=True  # Set to True to force reprocessing
    )
