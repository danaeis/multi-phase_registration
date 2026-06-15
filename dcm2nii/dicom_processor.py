import os
import sys
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import subprocess
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Optional, Dict
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline_logger import get_stage_logger

#from pipeline_logger import get_stage_logger 
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandardizedDICOMProcessor:
    """
    A class to process DICOM datasets with orientation correction, spacing standardization,
    and integration with existing DICOM traversal workflow.
    """
    
    def __init__(self, target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)):
        """
        Initialize the DICOM processor.
        
        Args:
            target_spacing: Target spacing in mm for (x, y, z) directions
        """
        self.target_spacing = target_spacing
        self.target_orientation = 'LPS'  # Left-Posterior-Superior
        
    def get_slice_thickness(self, series_path: str) -> float:
        """
        Get slice thickness from DICOM series (replicating your existing function).
        
        Args:
            series_path: Path to DICOM series directory
            
        Returns:
            Slice thickness value
        """
        try:
            dicom_files = [f for f in os.listdir(series_path) 
                          if f.lower().endswith(('.dcm', '.dicom')) or 
                          (os.path.isfile(os.path.join(series_path, f)) and not f.startswith('.'))]
            
            if not dicom_files:
                return float('inf')
            
            # Read first DICOM file
            first_file = os.path.join(series_path, dicom_files[0])
            ds = pydicom.dcmread(first_file)
            
            if hasattr(ds, 'SliceThickness') and ds.SliceThickness:
                return float(ds.SliceThickness)
            
            # Fallback: calculate from positions if available
            if len(dicom_files) > 1 and hasattr(ds, 'ImagePositionPatient'):
                positions = []
                for dcm_file in dicom_files[:10]:  # Sample first 10 files
                    try:
                        temp_ds = pydicom.dcmread(os.path.join(series_path, dcm_file))
                        if hasattr(temp_ds, 'ImagePositionPatient'):
                            positions.append(temp_ds.ImagePositionPatient[2])  # Z position
                    except:
                        continue
                
                if len(positions) > 1:
                    positions.sort()
                    thickness = abs(positions[1] - positions[0])
                    return thickness if thickness > 0 else float('inf')
            
            return float('inf')
            
        except Exception as e:
            logger.warning(f"Could not determine slice thickness for {series_path}: {e}")
            return float('inf')
    
    def load_and_standardize_dicom_series(self, dicom_path: str) -> sitk.Image:
        """
        Load a DICOM series and apply standardization (orientation + spacing).
        
        Args:
            dicom_path: Path to directory containing DICOM files
            
        Returns:
            Standardized SimpleITK Image object
        """
        logger.info(f"Loading and standardizing DICOM series from: {dicom_path}")
        
        try:
            # Load DICOM series
# Load DICOM series with multiple fallback methods
            image = self._load_dicom_with_fallbacks(dicom_path)
            
            # Debug: Print image statistics
            img_array = sitk.GetArrayFromImage(image)
            logger.info(f"Original - Size: {image.GetSize()}, Spacing: {image.GetSpacing()}")
            logger.info(f"Original - Pixel range: [{img_array.min():.2f}, {img_array.max():.2f}]")
            logger.info(f"Original - Mean: {img_array.mean():.2f}, Std: {img_array.std():.2f}")
            
            # Check if image has actual data
            if img_array.max() == img_array.min():
                logger.warning(f"Image appears to have constant values: {img_array.max()}")
            
            # Get orientation information from first DICOM file
            orientation_info = self.get_patient_orientation_info(dicom_path)
            
            # Apply orientation correction
            if 'image_orientation_patient' in orientation_info:
                current_orientation = self.determine_current_orientation(
                    orientation_info['image_orientation_patient']
                )
                image = self.reorient_to_lps(image, current_orientation)
            else:
                logger.warning("Image Orientation Patient tag not found, assuming already LPS")
            
            # Apply spacing standardization
            image = self.resample_to_target_spacing(image)
            
            # Final debug info
            final_array = sitk.GetArrayFromImage(image)
            logger.info(f"Standardized - Size: {image.GetSize()}, Spacing: {image.GetSpacing()}")
            logger.info(f"Standardized - Pixel range: [{final_array.min():.2f}, {final_array.max():.2f}]")
            
            return image
            
            
        except Exception as e:
            logger.error(f"Error processing DICOM series {dicom_path}: {str(e)}")
            raise
    
    def _load_dicom_with_fallbacks(self, dicom_path: str) -> sitk.Image:
        """
        Load DICOM series with multiple fallback methods to handle various DICOM formats.
        """
        # Method 1: Standard GDCM series reader
        try:
            logger.info("Trying method 1: GDCM series reader")
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(dicom_path)
            
            if series_ids:
                series_id = series_ids[0]
                dicom_names = reader.GetGDCMSeriesFileNames(dicom_path, series_id)
                
                if dicom_names:
                    reader.SetFileNames(dicom_names)
                    reader.MetaDataDictionaryArrayUpdateOn()
                    reader.LoadPrivateTagsOn()
                    
                    image = reader.Execute()
                    
                    # Apply DICOM pixel scaling if needed
                    image = self._apply_dicom_scaling(image, dicom_names[0])
                    
                    # Check if we got valid data
                    img_array = sitk.GetArrayFromImage(image)
                    if img_array.max() != img_array.min():
                        logger.info("Method 1 successful")
                        return image
                    else:
                        logger.warning("Method 1 produced constant values, trying method 2")
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
        
        # Method 2: Manual file sorting and reading
        try:
            logger.info("Trying method 2: Manual DICOM file reading")
            dicom_files = []
            for file in os.listdir(dicom_path):
                file_path = os.path.join(dicom_path, file)
                if os.path.isfile(file_path):
                    if file.lower().endswith(('.dcm', '.dicom')) or not file.startswith('.'):
                        try:
                            # Quick check if it's a valid DICOM
                            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                            if hasattr(ds, 'SOPClassUID'):
                                dicom_files.append(file_path)
                        except:
                            continue
            
            if not dicom_files:
                raise ValueError("No valid DICOM files found")
            
            # Sort by instance number or position if available
            def sort_key(file_path):
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    if hasattr(ds, 'InstanceNumber'):
                        return int(ds.InstanceNumber)
                    elif hasattr(ds, 'ImagePositionPatient'):
                        return float(ds.ImagePositionPatient[2])  # Z position
                    else:
                        return 0
                except:
                    return 0
            
            dicom_files.sort(key=sort_key)
            
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(dicom_files)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            
            image = reader.Execute()
            
            # Apply DICOM pixel scaling if needed
            image = self._apply_dicom_scaling(image, dicom_files[0])
            
            logger.info("Method 2 successful")
            return image
            
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
        
        # Method 3: Individual file reading and manual stacking
        try:
            logger.info("Trying method 3: Individual file stacking")
            return self._stack_individual_dicoms(dicom_path)
            
        except Exception as e:
            logger.error(f"All methods failed. Last error: {e}")
            raise ValueError(f"Could not load DICOM series from {dicom_path}")
    

    def _apply_dicom_scaling(self, image: sitk.Image, sample_dicom_path: str) -> sitk.Image:
        """
        Apply DICOM rescale slope and intercept if present.
        """
        # try:
        #     ds = pydicom.dcmread(sample_dicom_path)
            
        #     rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
        #     rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
            
        #     if rescale_slope != 1.0 or rescale_intercept != 0.0:
        #         logger.info(f"Applying DICOM scaling: slope={rescale_slope}, intercept={rescale_intercept}")
                
        #         # Apply scaling: output = slope * input + intercept
        #         scaled_image = sitk.Cast(image, sitk.sitkFloat32)
        #         scaled_image = scaled_image * rescale_slope + rescale_intercept
                
        #         # Convert back to appropriate type if needed
        #         if rescale_slope == 1.0 and rescale_intercept >= 0:
        #             if image.GetPixelID() in [sitk.sitkUInt8, sitk.sitkUInt16]:
        #                 scaled_image = sitk.Cast(scaled_image, image.GetPixelID())
                
        #         return scaled_image
            
        # except Exception as e:
        #     logger.warning(f"Could not apply DICOM scaling: {e}")
        
        return image
    
    def _stack_individual_dicoms(self, dicom_path: str) -> sitk.Image:
        """
        Last resort: manually read and stack individual DICOM files.
        """
        dicom_files = []
        slice_data = []
        
        # Collect all DICOM files
        for file in os.listdir(dicom_path):
            file_path = os.path.join(dicom_path, file)
            if os.path.isfile(file_path):
                try:
                    ds = pydicom.dcmread(file_path)
                    if hasattr(ds, 'pixel_array'):
                        z_pos = 0
                        if hasattr(ds, 'ImagePositionPatient'):
                            z_pos = float(ds.ImagePositionPatient[2])
                        elif hasattr(ds, 'InstanceNumber'):
                            z_pos = int(ds.InstanceNumber)
                        
                        slice_data.append((z_pos, file_path, ds))
                except:
                    continue
        
        if not slice_data:
            raise ValueError("No readable DICOM files found")
        
        # Sort by position/instance number
        slice_data.sort(key=lambda x: x[0])
        
        # Read pixel arrays and metadata
        pixel_arrays = []
        ref_ds = slice_data[0][2]
        
        for _, file_path, ds in slice_data:
            pixel_array = ds.pixel_array
            
            # Apply rescaling if present
            rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
            rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
            
            if rescale_slope != 1.0 or rescale_intercept != 0.0:
                pixel_array = pixel_array * rescale_slope + rescale_intercept
            
            pixel_arrays.append(pixel_array)
        
        # Stack into 3D array
        volume = np.stack(pixel_arrays, axis=0)
        
        # Create SimpleITK image
        image = sitk.GetImageFromArray(volume)
        
        # Set spacing
        spacing = [1.0, 1.0, 1.0]  # default
        if hasattr(ref_ds, 'PixelSpacing'):
            spacing[0] = float(ref_ds.PixelSpacing[0])  # x
            spacing[1] = float(ref_ds.PixelSpacing[1])  # y
        if hasattr(ref_ds, 'SliceThickness'):
            spacing[2] = float(ref_ds.SliceThickness)  # z
        elif len(slice_data) > 1:
            # Calculate from positions
            z_positions = [data[0] for data in slice_data]
            if len(set(z_positions)) > 1:  # Not all same position
                spacing[2] = abs(z_positions[1] - z_positions[0])
        
        image.SetSpacing(spacing)
        
        # Set origin
        if hasattr(ref_ds, 'ImagePositionPatient'):
            image.SetOrigin(ref_ds.ImagePositionPatient)
        
        logger.info(f"Manual stacking successful: {volume.shape}")
        return image
    

    def get_patient_orientation_info(self, dicom_directory: str) -> dict:
        """
        Extract patient orientation information from DICOM tags.
        """
        # Find DICOM files
        dicom_files = []
        for file in os.listdir(dicom_directory):
            file_path = os.path.join(dicom_directory, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(('.dcm', '.dicom')) or not file.startswith('.'):
                    dicom_files.append(file_path)
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_directory}")
        
        # Read the first DICOM file
        ds = pydicom.dcmread(dicom_files[0])
        
        orientation_info = {}
        
        # Image Position Patient (0020,0020)
        if hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
            orientation_info['image_position_patient'] = list(ds.ImagePositionPatient)
        
        # Image Orientation Patient (0020,0037)
        if hasattr(ds, 'ImageOrientationPatient') and ds.ImageOrientationPatient:
            orientation_info['image_orientation_patient'] = list(ds.ImageOrientationPatient)
        
        # Pixel Spacing
        if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
            orientation_info['pixel_spacing'] = list(ds.PixelSpacing)
        
        # Slice Thickness
        if hasattr(ds, 'SliceThickness'):
            orientation_info['slice_thickness'] = float(ds.SliceThickness)
        
        return orientation_info
    
    def determine_current_orientation(self, image_orientation_patient: List[float]) -> str:
        """
        Determine the current anatomical orientation from Image Orientation Patient.
        """
        if len(image_orientation_patient) != 6:
            raise ValueError("Image Orientation Patient must have 6 values")
        
        # Row direction cosines (first 3 values)
        row_cosines = np.array(image_orientation_patient[:3])
        # Column direction cosines (next 3 values)
        col_cosines = np.array(image_orientation_patient[3:])
        
        # Calculate slice direction (cross product)
        slice_cosines = np.cross(row_cosines, col_cosines)
        
        def get_dominant_axis(cosines):
            """Get the dominant axis and direction from direction cosines."""
            abs_cosines = np.abs(cosines)
            max_idx = np.argmax(abs_cosines)
            
            axes = ['x', 'y', 'z']
            axis = axes[max_idx]
            direction = '+' if cosines[max_idx] > 0 else '-'
            
            return axis, direction
        
        # Get orientation for each direction
        row_axis, row_dir = get_dominant_axis(row_cosines)
        col_axis, col_dir = get_dominant_axis(col_cosines)
        slice_axis, slice_dir = get_dominant_axis(slice_cosines)
        
        # Map to anatomical directions
        axis_map = {
            ('x', '+'): 'L', ('x', '-'): 'R',  # Left/Right
            ('y', '+'): 'P', ('y', '-'): 'A',  # Posterior/Anterior
            ('z', '+'): 'S', ('z', '-'): 'I'   # Superior/Inferior
        }
        
        row_anat = axis_map[(row_axis, row_dir)]
        col_anat = axis_map[(col_axis, col_dir)]
        slice_anat = axis_map[(slice_axis, slice_dir)]
        
        current_orientation = row_anat + col_anat + slice_anat
        logger.info(f"Current orientation: {current_orientation}")
        
        return current_orientation
    
    def reorient_to_lps(self, image: sitk.Image, current_orientation: str) -> sitk.Image:
        """
        Reorient the image to LPS orientation.
        """
        if current_orientation == self.target_orientation:
            logger.info("Image is already in LPS orientation")
            return image
        
        logger.info(f"Reorienting from {current_orientation} to {self.target_orientation}")
        reoriented_image = sitk.DICOMOrient(image, self.target_orientation)
        
        return reoriented_image
    
    # Standard CT air HU used as fill value for resampling boundaries.
    # DO NOT use image.min() — CT scanners store outside-FOV voxels as a sentinel
    # (raw -3024 → HU -4048 after RescaleIntercept=-1024). Using that sentinel as
    # SetDefaultPixelValue propagates -4048 into every resampled boundary voxel,
    # flooding the MMI histogram and causing registration to diverge.
    # Air in lungs is -1000 HU; -1024 is the standard "air/empty" fill.
    _RESAMPLE_FILL_HU: float = -1024.0

    # HU clipping range applied after resampling, before saving.
    # Lower -1024: removes the -4048 scanner sentinel (22% of voxels in this dataset)
    #   and resampling fill artefacts. No real anatomy exists below -1024 HU.
    # Upper +3000: covers cortical bone (700–1800 HU), calcified structures,
    #   and contrast-enhanced vessels (~+400 HU). Previous ceiling of 1024
    #   clipped dense bone, degrading TotalSegmentator bone segmentation.

    _CLIP_LOW_HU:  float = -1024.0
    _CLIP_HIGH_HU: float =  3000.0

    def resample_to_target_spacing(self, image: sitk.Image) -> sitk.Image:
        """
        Resample the image to the target spacing.

        Uses -1024 HU as boundary fill (standard CT air), not image.min() which
        is the scanner outside-FOV sentinel (-4048 HU).
        """
        original_spacing = image.GetSpacing()
        original_size    = image.GetSize()

        if np.allclose(original_spacing, self.target_spacing, atol=1e-3):
            logger.info("Image already has target spacing")
            return image

        new_size = [
            int(round(original_size[i] * original_spacing[i] / self.target_spacing[i]))
            for i in range(3)
        ]
        logger.info(f"Resampling from {original_spacing} to {self.target_spacing}")
        logger.info(f"Size change: {original_size} -> {new_size}")

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(self._RESAMPLE_FILL_HU)

        resampled_image = resampler.Execute(image)
        logger.info(f"Resampling complete. New spacing: {resampled_image.GetSpacing()}")
        return resampled_image

    def clip_hu_range(self, image: sitk.Image) -> sitk.Image:
        """
        Clip voxel values to [_CLIP_LOW_HU, _CLIP_HIGH_HU] and return Float32.

        Applied after resampling, before saving to NIfTI.
        Removes scanner outside-FOV sentinels (-4048 HU), resampling fill
        artefacts, and rare metal/noise spikes above +3000 HU.
        No real CT anatomy is lost within [-1024, +3000] HU.
        """
        arr = sitk.GetArrayFromImage(image)
        arr_clipped = np.clip(arr, self._CLIP_LOW_HU, self._CLIP_HIGH_HU).astype(np.float32)

        clipped = sitk.GetImageFromArray(arr_clipped)
        clipped.CopyInformation(image)

        n_low  = int((arr < self._CLIP_LOW_HU ).sum())
        n_high = int((arr > self._CLIP_HIGH_HU).sum())
        total  = arr.size
        logger.info(
            f"HU clipping [{self._CLIP_LOW_HU:.0f}, {self._CLIP_HIGH_HU:.0f}]: "
            f"removed {n_low} sentinel/fill voxels ({100*n_low/total:.2f}%) "
            f"and {n_high} high voxels ({100*n_high/total:.4f}%)"
        )
        return clipped
    
    def convert_standardized_dicom_to_nifti(
        self, 
        dicom_path: str, 
        output_path: str, 
        overwrite: bool = False
    ) -> None:
        """
        Convert DICOM series to standardized NIfTI format.
        
        Args:
            dicom_path: Path to the DICOM directory
            output_path: Path where the NIfTI file should be saved
            overwrite: Whether to overwrite existing files
        """
        # Parse study/series from the output filename stem so we can tag logs
        _stem = os.path.splitext(os.path.splitext(os.path.basename(output_path))[0])[0]
        _parts = _stem.split("_")
        _study_id  = _parts[0] if len(_parts) > 0 else "unknown"
        _series_id = _parts[1] if len(_parts) > 1 else "unknown"
        log = get_stage_logger("dcm2nii",
                               study_id=_study_id, series_id=_series_id)
        try:
            # Check if output already exists and overwrite flag
            if os.path.exists(output_path) and not overwrite:
                logger.info(f"Output already exists, skipping: {output_path}")
                return
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Load, standardize, and clip the DICOM series
            standardized_image = self.load_and_standardize_dicom_series(dicom_path)

            # Clip HU range to [-1024, +3000] — removes scanner outside-FOV
            # sentinels (-4048) and resampling fill artefacts before saving.
            # Must be done after resampling so the clipped value (-1024) is
            # consistent with the resampler fill value.
            standardized_image = self.clip_hu_range(standardized_image)

            # Save as NIfTI
            sitk.WriteImage(standardized_image, output_path)
            arr_final = sitk.GetArrayFromImage(standardized_image)
            log.info(f"Saved NIfTI: {output_path}")
            log.metric("conversion_result", {
                "status":        "ok",
                "output_path":   output_path,
                "spacing_mm":    list(standardized_image.GetSpacing()),
                "size":          list(standardized_image.GetSize()),
                "hu_min":        float(arr_final.min()),
                "hu_max":        float(arr_final.max()),
                "hu_mean":       round(float(arr_final.mean()), 1),
            })
            log.stage_summary({"status": "ok", "output_path": output_path})
            # Save processing metadata
            arr_final = sitk.GetArrayFromImage(standardized_image)
            metadata_path = output_path.replace('.nii.gz', '_metadata.txt')
            with open(metadata_path, 'w') as f:
                f.write(f"Original DICOM path: {dicom_path}\n")
                f.write(f"Target spacing: {self.target_spacing}\n")
                f.write(f"Final spacing: {standardized_image.GetSpacing()}\n")
                f.write(f"Final size: {standardized_image.GetSize()}\n")
                f.write(f"Target orientation: {self.target_orientation}\n")
                f.write(f"Final direction: {standardized_image.GetDirection()}\n")
                f.write(f"HU clip range: [{self._CLIP_LOW_HU:.0f}, {self._CLIP_HIGH_HU:.0f}]\n")
                f.write(f"Final HU min: {arr_final.min():.1f}\n")
                f.write(f"Final HU max: {arr_final.max():.1f}\n")
            
            logger.info(f"✓ Successfully converted and standardized {dicom_path} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing DICOM series {dicom_path}: {str(e)}")
            log.error(f"DICOM conversion failed: {str(e)}", exc_info=True)
            log.metric("conversion_result", {
                "status":      "failed",
                "dicom_path":  dicom_path,
                "error":       str(e),
            })
            log.stage_summary({"status": "failed", "error": str(e)})
            raise

# Modified version of your functions to integrate standardization
def save_dicom_paths_with_standardization(
    batch_dir: str, 
    labels_csv: str, 
    output_pkl: str,
    processor: StandardizedDICOMProcessor
) -> List[Dict]:
    """
    Select one optimal series per phase per study and prepare for standardization.
    Modified version that uses the processor's slice thickness function.
    """
    labels_df = pd.read_csv(labels_csv)
    phase_lookup = {
        (row['StudyInstanceUID'], row['SeriesInstanceUID']): row['Label'].lower()
        for _, row in labels_df.iterrows()
    }

    case_count = 0
    best_series = defaultdict(lambda: (float('inf'), None))
    failed_count = 0
    
    print(f"Processing batches in {batch_dir}...")
    
    for batch in os.listdir(batch_dir):
        batch_path = os.path.join(batch_dir, batch)
        if not os.path.isdir(batch_path):
            continue
            
        for study in os.listdir(batch_path):
            study_path = os.path.join(batch_path, study)
            if not os.path.isdir(study_path):
                continue
                
            case_count += 1
            
            for series in os.listdir(study_path):
                series_path = os.path.join(study_path, series)
                if not os.path.isdir(series_path):
                    continue
                    
                key = (study, series)
                
                if key not in phase_lookup:
                    continue
                
                phase = phase_lookup[key]
                thickness = processor.get_slice_thickness(series_path)
                
                if thickness == float('inf'):
                    failed_count += 1
                    print("inf thickness")
                    continue
                
                current_best_thickness = best_series[(study, phase)][0]
                print("thickness, current_best_thickness", thickness, " ", current_best_thickness)
                
                if thickness < current_best_thickness:
                    best_series[(study, phase)] = (
                        thickness,
                        {
                            'study_uid': study,
                            'series_uid': series,
                            'series_path': series_path,
                            'phase': phase,
                            'slice_thickness': thickness
                        }
                    )

    # Extract results
    final_series_data = [info for (thickness, info) in best_series.values() if info is not None]
    
    # Save results
    with open(output_pkl, 'wb') as f:
        pickle.dump(final_series_data, f)
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Total cases processed: {case_count}")
    print(f"Series with infinite thickness: {failed_count}")
    print(f"Final series selected: {len(final_series_data)}")
    print(f"Success rate: {((case_count - failed_count) / case_count * 100):.1f}%")
    
    return final_series_data

def process_series_list_to_standardized_nifti(
    series_data: List[Dict], 
    output_base_dir: str,
    processor: StandardizedDICOMProcessor,
    overwrite: bool = False
) -> None:
    """
    Process a list of DICOM series to standardized NIfTI files.
    
    Args:
        series_data: List of series dictionaries from save_dicom_paths
        output_base_dir: Base directory for output files
        processor: StandardizedDICOMProcessor instance
        overwrite: Whether to overwrite existing files
    """
    successful_conversions = 0
    failed_conversions = 0
    
    # print(f"\nProcessing {len(series_data)} series to standardized NIfTI format...")
    
    for i, series_info in enumerate(series_data, 1):
        try:
            study_uid = series_info['study_uid']
            series_uid = series_info['series_uid']
            series_path = series_info['series_path']
            phase = series_info['phase']
            
            # Create output filename
            output_filename = f"{study_uid}_{series_uid}_standardized.nii.gz"
            output_path = os.path.join(output_base_dir, output_filename)
            
            # print(f"[{i}/{len(series_data)}] Processing: {study_uid}_{phase}")
            
            # Convert with standardization
            processor.convert_standardized_dicom_to_nifti(
                dicom_path=series_path,
                output_path=output_path,
                overwrite=overwrite
            )
            
            successful_conversions += 1
            
        except Exception as e:
            print(f"✗ Failed to process series {series_info.get('series_uid', 'unknown')}: {str(e)}")
            failed_conversions += 1
            continue
    
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Success rate: {(successful_conversions / len(series_data) * 100):.1f}%")

# Complete workflow function
def complete_standardization_workflow(
    batch_dir: str,
    labels_csv: str,
    output_base_dir: str,
    pkl_path: str = "dicom_paths.pkl",
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    overwrite: bool = False
) -> None:
    """
    Complete workflow: traverse dataset, select optimal series, and convert to standardized NIfTI.
    
    Args:
        batch_dir: Directory containing batch subdirectories
        labels_csv: CSV file with labels
        output_base_dir: Base directory for output files
        target_spacing: Target spacing for standardization
        overwrite: Whether to overwrite existing files
    """
    # Initialize processor
    processor = StandardizedDICOMProcessor(target_spacing=target_spacing)
    
    # Create temporary pickle file for series data
    if pkl_path is None:
        temp_pkl = os.path.join(output_base_dir, "temp_series_data.pkl")
    os.makedirs(output_base_dir, exist_ok=True)

    if not os.path.exists(output_base_dir) or not os.listdir(output_base_dir) or overwrite:
        print("NIfTI directory empty/overwrite requested. Processing DICOMs...")
        
        # Load or generate DICOM paths cache
        if os.path.exists(pkl_path) and not overwrite:
            with open(pkl_path, 'rb') as f:
                series_data = pickle.load(f)
            print(f"Loaded {len(series_data)} series from cache")
        else:
            print("Starting to load DICOM series with metadata...")
            series_data = save_dicom_paths_with_standardization(batch_dir, labels_csv, pkl_path, processor)
            # series_data = load_and_cache_dicom_series(batch_dir, labels_csv, pkl_path)
        print("converting")
        # Convert to NIfTI
        process_series_list_to_standardized_nifti(
            series_data, 
            output_base_dir, 
            processor,
            overwrite
        )
    else:
        print(f"NIfTI files already exist in {output_base_dir}. Skipping conversion.")
