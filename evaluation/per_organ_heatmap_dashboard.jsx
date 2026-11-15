import React, { useMemo, useState } from "react";

const DATA = {
 "generated_note": "Computed from all_conditions.csv",
 "matched_set_size": 26,
 "organs": [
  "liver",
  "spleen",
  "kidney_right",
  "kidney_left",
  "aorta",
  "inferior_vena_cava",
  "vertebrae_L1",
  "vertebrae_L2",
  "vertebrae_L3",
  "pancreas",
  "gallbladder"
 ],
 "groups": {
  "ablation_aligned": {
   "label": "Ablation (aligned)",
   "conditions": [
    "A1_crop_only",
    "A2_zalign",
    "A3_pass0",
    "A4_pass01",
    "A5_pass012",
    "A6_bspline"
   ]
  },
  "ablation_noalign": {
   "label": "Ablation (no-align)",
   "conditions": [
    "A3b_pass0_noalign",
    "A4b_pass01_noalign",
    "A5b_pass012_noalign",
    "A6_bspline_baseline"
   ]
  },
  "r_series": {
   "label": "R-series",
   "conditions": [
    "R_organ__aligned",
    "R_sobel__aligned",
    "R_full__aligned",
    "R_full__baseline",
    "R_full__raw",
    "R_mind__aligned",
    "R_mind__baseline",
    "R_mind__raw"
   ]
  },
  "baselines": {
   "label": "Baselines",
   "conditions": [
    "B2_deeds__aligned",
    "B2_deeds__baseline",
    "B2_deeds__raw",
    "B3_ants__aligned",
    "B3_ants__baseline",
    "B3_ants__raw",
    "B5_unigradicon__aligned",
    "B5_unigradicon__baseline",
    "B5_unigradicon__raw",
    "B5_unigradicon_io__aligned",
    "B5_unigradicon_io__baseline",
    "B5_unigradicon_io__raw"
   ]
  }
 },
 "conditions": {
  "A1_crop_only": {
   "condition": "A1_crop_only",
   "group": "ablation_aligned",
   "status": "ok",
   "n_studies": 23,
   "n_studies_total_available": 24,
   "n_studies_matched_available": 23,
   "overall": {
    "dice_med": 0.797,
    "dice_q1": 0.62,
    "dice_q3": 0.894,
    "centroid_med": 2.629,
    "hd95_med": 3.354,
    "n_rows": 498
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.923,
     "centroid_med": 3.052,
     "hd95_med": 4.743,
     "n": 47
    },
    "spleen": {
     "dice_med": 0.85,
     "centroid_med": 3.355,
     "hd95_med": 4.307,
     "n": 47
    },
    "kidney_right": {
     "dice_med": 0.777,
     "centroid_med": 2.753,
     "hd95_med": 3.674,
     "n": 47
    },
    "kidney_left": {
     "dice_med": 0.727,
     "centroid_med": 3.175,
     "hd95_med": 4.243,
     "n": 47
    },
    "aorta": {
     "dice_med": 0.753,
     "centroid_med": 3.214,
     "hd95_med": 3.354,
     "n": 47
    },
    "inferior_vena_cava": {
     "dice_med": 0.507,
     "centroid_med": 6.303,
     "hd95_med": 4.743,
     "n": 47
    },
    "vertebrae_L1": {
     "dice_med": 0.898,
     "centroid_med": 1.077,
     "hd95_med": 1.5,
     "n": 47
    },
    "vertebrae_L2": {
     "dice_med": 0.905,
     "centroid_med": 0.991,
     "hd95_med": 1.5,
     "n": 44
    },
    "vertebrae_L3": {
     "dice_med": 0.912,
     "centroid_med": 1.005,
     "hd95_med": 1.5,
     "n": 41
    },
    "pancreas": {
     "dice_med": 0.529,
     "centroid_med": 3.249,
     "hd95_med": 5.196,
     "n": 47
    },
    "gallbladder": {
     "dice_med": 0.497,
     "centroid_med": 4.597,
     "hd95_med": 4.743,
     "n": 37
    }
   },
   "flag": null
  },
  "A2_zalign": {
   "condition": "A2_zalign",
   "group": "ablation_aligned",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.773,
    "dice_q1": 0.573,
    "dice_q3": 0.881,
    "centroid_med": 2.92,
    "hd95_med": 3.674,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.923,
     "centroid_med": 3.056,
     "hd95_med": 6.0,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.851,
     "centroid_med": 3.291,
     "hd95_med": 4.5,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.778,
     "centroid_med": 2.753,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.718,
     "centroid_med": 3.542,
     "hd95_med": 4.243,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.757,
     "centroid_med": 2.985,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.517,
     "centroid_med": 5.684,
     "hd95_med": 5.196,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.838,
     "centroid_med": 1.408,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.885,
     "centroid_med": 1.235,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.895,
     "centroid_med": 1.243,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.563,
     "centroid_med": 3.637,
     "hd95_med": 4.975,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.486,
     "centroid_med": 4.905,
     "hd95_med": 4.975,
     "n": 42
    }
   },
   "flag": null
  },
  "A3_pass0": {
   "condition": "A3_pass0",
   "group": "ablation_aligned",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.792,
    "dice_q1": 0.624,
    "dice_q3": 0.89,
    "centroid_med": 2.288,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.927,
     "centroid_med": 2.163,
     "hd95_med": 4.743,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.881,
     "centroid_med": 2.271,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.789,
     "centroid_med": 2.047,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.749,
     "centroid_med": 2.497,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.729,
     "centroid_med": 2.534,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.567,
     "centroid_med": 5.136,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.87,
     "centroid_med": 1.296,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.893,
     "centroid_med": 1.193,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.887,
     "centroid_med": 1.095,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.572,
     "centroid_med": 2.754,
     "hd95_med": 4.975,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.597,
     "centroid_med": 4.321,
     "hd95_med": 4.5,
     "n": 42
    }
   },
   "flag": null
  },
  "A4_pass01": {
   "condition": "A4_pass01",
   "group": "ablation_aligned",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.816,
    "dice_q1": 0.632,
    "dice_q3": 0.902,
    "centroid_med": 2.116,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.928,
     "centroid_med": 2.379,
     "hd95_med": 4.975,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.877,
     "centroid_med": 2.332,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.79,
     "centroid_med": 1.974,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.757,
     "centroid_med": 2.49,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.734,
     "centroid_med": 2.629,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.559,
     "centroid_med": 6.683,
     "hd95_med": 4.975,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.91,
     "centroid_med": 0.9,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.903,
     "centroid_med": 0.826,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.91,
     "centroid_med": 0.787,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.541,
     "centroid_med": 3.208,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.578,
     "centroid_med": 4.34,
     "hd95_med": 4.743,
     "n": 42
    }
   },
   "flag": null
  },
  "A5_pass012": {
   "condition": "A5_pass012",
   "group": "ablation_aligned",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.816,
    "dice_q1": 0.632,
    "dice_q3": 0.902,
    "centroid_med": 2.116,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.928,
     "centroid_med": 2.379,
     "hd95_med": 4.975,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.877,
     "centroid_med": 2.332,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.79,
     "centroid_med": 1.974,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.757,
     "centroid_med": 2.49,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.734,
     "centroid_med": 2.629,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.559,
     "centroid_med": 6.683,
     "hd95_med": 4.975,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.91,
     "centroid_med": 0.9,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.903,
     "centroid_med": 0.826,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.91,
     "centroid_med": 0.787,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.541,
     "centroid_med": 3.208,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.578,
     "centroid_med": 4.34,
     "hd95_med": 4.743,
     "n": 42
    }
   },
   "flag": "Numerically identical to A4_pass01 (Pass 2 universally rejected by QC)"
  },
  "A6_bspline": {
   "condition": "A6_bspline",
   "group": "ablation_aligned",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.816,
    "dice_q1": 0.632,
    "dice_q3": 0.902,
    "centroid_med": 2.116,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.928,
     "centroid_med": 2.379,
     "hd95_med": 4.975,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.877,
     "centroid_med": 2.332,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.79,
     "centroid_med": 1.974,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.757,
     "centroid_med": 2.49,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.734,
     "centroid_med": 2.629,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.559,
     "centroid_med": 6.683,
     "hd95_med": 4.975,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.91,
     "centroid_med": 0.9,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.903,
     "centroid_med": 0.826,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.91,
     "centroid_med": 0.787,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.541,
     "centroid_med": 3.208,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.578,
     "centroid_med": 4.34,
     "hd95_med": 4.743,
     "n": 42
    }
   },
   "flag": "Numerically identical to A4_pass01 (B-spline evaluation not yet valid)"
  },
  "A3b_pass0_noalign": {
   "condition": "A3b_pass0_noalign",
   "group": "ablation_noalign",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.797,
    "dice_q1": 0.631,
    "dice_q3": 0.89,
    "centroid_med": 2.278,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.927,
     "centroid_med": 2.11,
     "hd95_med": 4.743,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.877,
     "centroid_med": 2.334,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.783,
     "centroid_med": 1.974,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.749,
     "centroid_med": 2.531,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.749,
     "centroid_med": 2.534,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.567,
     "centroid_med": 5.346,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.876,
     "centroid_med": 1.224,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.889,
     "centroid_med": 1.193,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.887,
     "centroid_med": 1.095,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.553,
     "centroid_med": 2.889,
     "hd95_med": 4.975,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.604,
     "centroid_med": 4.432,
     "hd95_med": 4.743,
     "n": 42
    }
   },
   "flag": null
  },
  "A4b_pass01_noalign": {
   "condition": "A4b_pass01_noalign",
   "group": "ablation_noalign",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.821,
    "dice_q1": 0.651,
    "dice_q3": 0.903,
    "centroid_med": 2.098,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.922,
     "centroid_med": 2.484,
     "hd95_med": 4.975,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.88,
     "centroid_med": 2.357,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.79,
     "centroid_med": 1.934,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.761,
     "centroid_med": 2.604,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.74,
     "centroid_med": 2.802,
     "hd95_med": 3.0,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.572,
     "centroid_med": 6.555,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.901,
     "centroid_med": 0.9,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.908,
     "centroid_med": 0.809,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.914,
     "centroid_med": 0.815,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.541,
     "centroid_med": 3.025,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.579,
     "centroid_med": 4.542,
     "hd95_med": 4.743,
     "n": 42
    }
   },
   "flag": null
  },
  "A5b_pass012_noalign": {
   "condition": "A5b_pass012_noalign",
   "group": "ablation_noalign",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.821,
    "dice_q1": 0.651,
    "dice_q3": 0.903,
    "centroid_med": 2.098,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.922,
     "centroid_med": 2.484,
     "hd95_med": 4.975,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.88,
     "centroid_med": 2.357,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.79,
     "centroid_med": 1.934,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.761,
     "centroid_med": 2.604,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.74,
     "centroid_med": 2.802,
     "hd95_med": 3.0,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.572,
     "centroid_med": 6.555,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.901,
     "centroid_med": 0.9,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.908,
     "centroid_med": 0.809,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.914,
     "centroid_med": 0.815,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.541,
     "centroid_med": 3.025,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.579,
     "centroid_med": 4.542,
     "hd95_med": 4.743,
     "n": 42
    }
   },
   "flag": null
  },
  "A6_bspline_baseline": {
   "condition": "A6_bspline_baseline",
   "group": "ablation_noalign",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.821,
    "dice_q1": 0.651,
    "dice_q3": 0.903,
    "centroid_med": 2.098,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.922,
     "centroid_med": 2.484,
     "hd95_med": 4.975,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.88,
     "centroid_med": 2.357,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.79,
     "centroid_med": 1.934,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.761,
     "centroid_med": 2.604,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.74,
     "centroid_med": 2.802,
     "hd95_med": 3.0,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.572,
     "centroid_med": 6.555,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.901,
     "centroid_med": 0.9,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.908,
     "centroid_med": 0.809,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.914,
     "centroid_med": 0.815,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.541,
     "centroid_med": 3.025,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.579,
     "centroid_med": 4.542,
     "hd95_med": 4.743,
     "n": 42
    }
   },
   "flag": null
  },
  "R_organ__aligned": {
   "condition": "R_organ__aligned",
   "group": "r_series",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.793,
    "dice_q1": 0.632,
    "dice_q3": 0.898,
    "centroid_med": 1.95,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.935,
     "centroid_med": 1.506,
     "hd95_med": 3.674,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.902,
     "centroid_med": 1.428,
     "hd95_med": 3.0,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.824,
     "centroid_med": 1.341,
     "hd95_med": 3.0,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.791,
     "centroid_med": 1.73,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.724,
     "centroid_med": 2.508,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.582,
     "centroid_med": 4.939,
     "hd95_med": 4.5,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.86,
     "centroid_med": 1.472,
     "hd95_med": 2.121,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.889,
     "centroid_med": 1.54,
     "hd95_med": 2.121,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.869,
     "centroid_med": 1.307,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.601,
     "centroid_med": 2.789,
     "hd95_med": 4.743,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.616,
     "centroid_med": 3.944,
     "hd95_med": 4.5,
     "n": 42
    }
   },
   "flag": null
  },
  "R_sobel__aligned": {
   "condition": "R_sobel__aligned",
   "group": "r_series",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.761,
    "dice_q1": 0.567,
    "dice_q3": 0.886,
    "centroid_med": 2.932,
    "hd95_med": 3.674,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.919,
     "centroid_med": 2.932,
     "hd95_med": 4.743,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.86,
     "centroid_med": 3.201,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.743,
     "centroid_med": 2.489,
     "hd95_med": 3.674,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.742,
     "centroid_med": 3.304,
     "hd95_med": 4.243,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.713,
     "centroid_med": 2.953,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.577,
     "centroid_med": 5.26,
     "hd95_med": 4.975,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.871,
     "centroid_med": 1.274,
     "hd95_med": 2.121,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.892,
     "centroid_med": 1.191,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.899,
     "centroid_med": 1.201,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.545,
     "centroid_med": 3.995,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.525,
     "centroid_med": 4.845,
     "hd95_med": 5.408,
     "n": 42
    }
   },
   "flag": null
  },
  "R_full__aligned": {
   "condition": "R_full__aligned",
   "group": "r_series",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.805,
    "dice_q1": 0.634,
    "dice_q3": 0.897,
    "centroid_med": 1.986,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.932,
     "centroid_med": 1.813,
     "hd95_med": 4.243,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.897,
     "centroid_med": 1.693,
     "hd95_med": 3.0,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.82,
     "centroid_med": 1.479,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.786,
     "centroid_med": 1.888,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.738,
     "centroid_med": 2.328,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.587,
     "centroid_med": 4.78,
     "hd95_med": 4.5,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.881,
     "centroid_med": 1.291,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.892,
     "centroid_med": 1.223,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.893,
     "centroid_med": 1.127,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.61,
     "centroid_med": 2.759,
     "hd95_med": 4.975,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.6,
     "centroid_med": 4.042,
     "hd95_med": 4.5,
     "n": 42
    }
   },
   "flag": null
  },
  "R_full__baseline": {
   "condition": "R_full__baseline",
   "group": "r_series",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.806,
    "dice_q1": 0.636,
    "dice_q3": 0.899,
    "centroid_med": 1.986,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.932,
     "centroid_med": 1.813,
     "hd95_med": 4.243,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.899,
     "centroid_med": 1.693,
     "hd95_med": 3.0,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.819,
     "centroid_med": 1.483,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.786,
     "centroid_med": 1.66,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.744,
     "centroid_med": 2.328,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.579,
     "centroid_med": 4.612,
     "hd95_med": 4.5,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.882,
     "centroid_med": 1.279,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.892,
     "centroid_med": 1.134,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.897,
     "centroid_med": 1.084,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.606,
     "centroid_med": 2.755,
     "hd95_med": 4.975,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.607,
     "centroid_med": 3.837,
     "hd95_med": 4.5,
     "n": 42
    }
   },
   "flag": null
  },
  "R_full__raw": {
   "condition": "R_full__raw",
   "group": "r_series",
   "status": "partial",
   "n_studies": 2,
   "n_studies_total_available": 2,
   "n_studies_matched_available": 2,
   "overall": {
    "dice_med": 0.626,
    "dice_q1": 0.257,
    "dice_q3": 0.8,
    "centroid_med": 4.996,
    "hd95_med": 7.019,
    "n_rows": 21
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.846,
     "centroid_med": 2.671,
     "hd95_med": 8.58,
     "n": 2
    },
    "spleen": {
     "dice_med": 0.812,
     "centroid_med": 2.928,
     "hd95_med": 5.294,
     "n": 2
    },
    "kidney_right": {
     "dice_med": 0.684,
     "centroid_med": 2.758,
     "hd95_med": 6.16,
     "n": 2
    },
    "kidney_left": {
     "dice_med": 0.642,
     "centroid_med": 1.654,
     "hd95_med": 6.924,
     "n": 2
    },
    "aorta": {
     "dice_med": 0.337,
     "centroid_med": 13.701,
     "hd95_med": 15.434,
     "n": 2
    },
    "inferior_vena_cava": {
     "dice_med": 0.175,
     "centroid_med": 16.501,
     "hd95_med": 13.643,
     "n": 2
    },
    "vertebrae_L1": {
     "dice_med": 0.429,
     "centroid_med": 12.053,
     "hd95_med": 11.593,
     "n": 2
    },
    "vertebrae_L2": {
     "dice_med": 0.444,
     "centroid_med": 11.199,
     "hd95_med": 11.773,
     "n": 2
    },
    "vertebrae_L3": {
     "dice_med": 0.416,
     "centroid_med": 11.639,
     "hd95_med": 12.311,
     "n": 2
    },
    "pancreas": {
     "dice_med": 0.307,
     "centroid_med": 7.178,
     "hd95_med": 8.3,
     "n": 2
    },
    "gallbladder": {
     "dice_med": 0.41,
     "centroid_med": 8.579,
     "hd95_med": 9.792,
     "n": 1
    }
   },
   "flag": null
  },
  "R_mind__aligned": {
   "condition": "R_mind__aligned",
   "group": "r_series",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.798,
    "dice_q1": 0.606,
    "dice_q3": 0.899,
    "centroid_med": 2.409,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.922,
     "centroid_med": 2.7,
     "hd95_med": 4.5,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.896,
     "centroid_med": 2.349,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.781,
     "centroid_med": 1.995,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.772,
     "centroid_med": 2.312,
     "hd95_med": 3.674,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.726,
     "centroid_med": 2.889,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.553,
     "centroid_med": 5.351,
     "hd95_med": 4.975,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.881,
     "centroid_med": 1.189,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.905,
     "centroid_med": 1.036,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.909,
     "centroid_med": 1.044,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.57,
     "centroid_med": 3.259,
     "hd95_med": 4.975,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.572,
     "centroid_med": 4.75,
     "hd95_med": 5.408,
     "n": 42
    }
   },
   "flag": null
  },
  "R_mind__baseline": {
   "condition": "R_mind__baseline",
   "group": "r_series",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.799,
    "dice_q1": 0.621,
    "dice_q3": 0.898,
    "centroid_med": 2.366,
    "hd95_med": 3.354,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.921,
     "centroid_med": 2.906,
     "hd95_med": 4.5,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.889,
     "centroid_med": 2.274,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.792,
     "centroid_med": 2.098,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.755,
     "centroid_med": 2.178,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.726,
     "centroid_med": 2.989,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.569,
     "centroid_med": 5.351,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.881,
     "centroid_med": 1.215,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.903,
     "centroid_med": 1.069,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.907,
     "centroid_med": 1.109,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.563,
     "centroid_med": 3.319,
     "hd95_med": 4.975,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.525,
     "centroid_med": 5.215,
     "hd95_med": 5.408,
     "n": 42
    }
   },
   "flag": null
  },
  "R_mind__raw": {
   "condition": "R_mind__raw",
   "group": "r_series",
   "status": "ok",
   "n_studies": 18,
   "n_studies_total_available": 21,
   "n_studies_matched_available": 18,
   "overall": {
    "dice_med": 0.425,
    "dice_q1": 0.066,
    "dice_q3": 0.678,
    "centroid_med": 10.324,
    "hd95_med": 10.062,
    "n_rows": 300
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.799,
     "centroid_med": 10.54,
     "hd95_med": 9.604,
     "n": 28
    },
    "spleen": {
     "dice_med": 0.554,
     "centroid_med": 9.332,
     "hd95_med": 10.816,
     "n": 28
    },
    "kidney_right": {
     "dice_med": 0.447,
     "centroid_med": 11.158,
     "hd95_med": 9.243,
     "n": 28
    },
    "kidney_left": {
     "dice_med": 0.397,
     "centroid_med": 10.987,
     "hd95_med": 10.334,
     "n": 28
    },
    "aorta": {
     "dice_med": 0.233,
     "centroid_med": 13.155,
     "hd95_med": 14.951,
     "n": 28
    },
    "inferior_vena_cava": {
     "dice_med": 0.099,
     "centroid_med": 13.5,
     "hd95_med": 11.221,
     "n": 28
    },
    "vertebrae_L1": {
     "dice_med": 0.326,
     "centroid_med": 10.261,
     "hd95_med": 8.745,
     "n": 28
    },
    "vertebrae_L2": {
     "dice_med": 0.376,
     "centroid_med": 10.099,
     "hd95_med": 8.682,
     "n": 27
    },
    "vertebrae_L3": {
     "dice_med": 0.416,
     "centroid_med": 9.235,
     "hd95_med": 8.412,
     "n": 28
    },
    "pancreas": {
     "dice_med": 0.226,
     "centroid_med": 9.952,
     "hd95_med": 10.712,
     "n": 28
    },
    "gallbladder": {
     "dice_med": 0.187,
     "centroid_med": 9.215,
     "hd95_med": 8.617,
     "n": 21
    }
   },
   "flag": null
  },
  "B2_deeds__aligned": {
   "condition": "B2_deeds__aligned",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.886,
    "dice_q1": 0.781,
    "dice_q3": 0.931,
    "centroid_med": 1.014,
    "hd95_med": 2.121,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.961,
     "centroid_med": 0.927,
     "hd95_med": 2.598,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.936,
     "centroid_med": 0.989,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.875,
     "centroid_med": 0.76,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.877,
     "centroid_med": 0.759,
     "hd95_med": 2.121,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.842,
     "centroid_med": 1.681,
     "hd95_med": 2.121,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.674,
     "centroid_med": 5.2,
     "hd95_med": 3.674,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.911,
     "centroid_med": 0.751,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.923,
     "centroid_med": 0.669,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.93,
     "centroid_med": 0.631,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.714,
     "centroid_med": 1.954,
     "hd95_med": 4.243,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.787,
     "centroid_med": 1.412,
     "hd95_med": 2.598,
     "n": 42
    }
   },
   "flag": null
  },
  "B2_deeds__baseline": {
   "condition": "B2_deeds__baseline",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.882,
    "dice_q1": 0.777,
    "dice_q3": 0.926,
    "centroid_med": 1.076,
    "hd95_med": 2.121,
    "n_rows": 577
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.961,
     "centroid_med": 0.946,
     "hd95_med": 2.598,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.937,
     "centroid_med": 1.022,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.869,
     "centroid_med": 0.769,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.873,
     "centroid_med": 0.787,
     "hd95_med": 2.121,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.838,
     "centroid_med": 2.118,
     "hd95_med": 2.121,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.699,
     "centroid_med": 3.875,
     "hd95_med": 3.354,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.908,
     "centroid_med": 0.847,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.911,
     "centroid_med": 0.78,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.923,
     "centroid_med": 0.78,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.704,
     "centroid_med": 2.081,
     "hd95_med": 4.243,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.781,
     "centroid_med": 1.604,
     "hd95_med": 2.121,
     "n": 42
    }
   },
   "flag": null
  },
  "B2_deeds__raw": {
   "condition": "B2_deeds__raw",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.88,
    "dice_q1": 0.757,
    "dice_q3": 0.926,
    "centroid_med": 1.171,
    "hd95_med": 2.121,
    "n_rows": 582
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.958,
     "centroid_med": 1.071,
     "hd95_med": 3.0,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.935,
     "centroid_med": 1.092,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.866,
     "centroid_med": 0.998,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.864,
     "centroid_med": 0.838,
     "hd95_med": 2.121,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.8,
     "centroid_med": 3.635,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.648,
     "centroid_med": 5.385,
     "hd95_med": 4.243,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.913,
     "centroid_med": 0.859,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.917,
     "centroid_med": 0.775,
     "hd95_med": 1.5,
     "n": 51
    },
    "vertebrae_L3": {
     "dice_med": 0.922,
     "centroid_med": 0.735,
     "hd95_med": 1.5,
     "n": 49
    },
    "pancreas": {
     "dice_med": 0.709,
     "centroid_med": 2.051,
     "hd95_med": 4.243,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.792,
     "centroid_med": 1.618,
     "hd95_med": 2.598,
     "n": 42
    }
   },
   "flag": null
  },
  "B3_ants__aligned": {
   "condition": "B3_ants__aligned",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.809,
    "dice_q1": 0.665,
    "dice_q3": 0.916,
    "centroid_med": 1.584,
    "hd95_med": 3.0,
    "n_rows": 574
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.945,
     "centroid_med": 1.515,
     "hd95_med": 3.674,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.907,
     "centroid_med": 1.348,
     "hd95_med": 2.598,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.783,
     "centroid_med": 1.59,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.769,
     "centroid_med": 1.793,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.705,
     "centroid_med": 2.459,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.651,
     "centroid_med": 5.015,
     "hd95_med": 4.243,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.905,
     "centroid_med": 0.813,
     "hd95_med": 1.5,
     "n": 54
    },
    "vertebrae_L2": {
     "dice_med": 0.911,
     "centroid_med": 0.659,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.925,
     "centroid_med": 0.681,
     "hd95_med": 1.5,
     "n": 44
    },
    "pancreas": {
     "dice_med": 0.596,
     "centroid_med": 3.085,
     "hd95_med": 4.743,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.654,
     "centroid_med": 3.122,
     "hd95_med": 4.243,
     "n": 42
    }
   },
   "flag": null
  },
  "B3_ants__baseline": {
   "condition": "B3_ants__baseline",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.808,
    "dice_q1": 0.665,
    "dice_q3": 0.916,
    "centroid_med": 1.543,
    "hd95_med": 3.0,
    "n_rows": 578
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.946,
     "centroid_med": 1.532,
     "hd95_med": 3.674,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.904,
     "centroid_med": 1.453,
     "hd95_med": 2.598,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.784,
     "centroid_med": 1.556,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.773,
     "centroid_med": 1.796,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.716,
     "centroid_med": 2.22,
     "hd95_med": 3.354,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.649,
     "centroid_med": 5.409,
     "hd95_med": 4.243,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.903,
     "centroid_med": 0.773,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.915,
     "centroid_med": 0.653,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.925,
     "centroid_med": 0.689,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.602,
     "centroid_med": 2.833,
     "hd95_med": 4.743,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.651,
     "centroid_med": 3.432,
     "hd95_med": 4.243,
     "n": 43
    }
   },
   "flag": null
  },
  "B3_ants__raw": {
   "condition": "B3_ants__raw",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.817,
    "dice_q1": 0.647,
    "dice_q3": 0.919,
    "centroid_med": 1.63,
    "hd95_med": 3.354,
    "n_rows": 582
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.946,
     "centroid_med": 1.398,
     "hd95_med": 3.674,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.905,
     "centroid_med": 1.403,
     "hd95_med": 2.598,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.788,
     "centroid_med": 1.562,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.766,
     "centroid_med": 2.19,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.657,
     "centroid_med": 7.438,
     "hd95_med": 6.0,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.617,
     "centroid_med": 8.095,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.907,
     "centroid_med": 0.552,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.921,
     "centroid_med": 0.488,
     "hd95_med": 1.5,
     "n": 51
    },
    "vertebrae_L3": {
     "dice_med": 0.925,
     "centroid_med": 0.542,
     "hd95_med": 1.5,
     "n": 49
    },
    "pancreas": {
     "dice_med": 0.607,
     "centroid_med": 2.882,
     "hd95_med": 4.743,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.652,
     "centroid_med": 3.23,
     "hd95_med": 4.243,
     "n": 42
    }
   },
   "flag": null
  },
  "B5_unigradicon__aligned": {
   "condition": "B5_unigradicon__aligned",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.794,
    "dice_q1": 0.627,
    "dice_q3": 0.911,
    "centroid_med": 1.352,
    "hd95_med": 3.0,
    "n_rows": 578
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.949,
     "centroid_med": 1.257,
     "hd95_med": 3.354,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.923,
     "centroid_med": 1.202,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.716,
     "centroid_med": 1.213,
     "hd95_med": 3.0,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.737,
     "centroid_med": 1.423,
     "hd95_med": 2.598,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.651,
     "centroid_med": 6.149,
     "hd95_med": 4.243,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.618,
     "centroid_med": 5.991,
     "hd95_med": 4.5,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.884,
     "centroid_med": 0.874,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.897,
     "centroid_med": 0.791,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.913,
     "centroid_med": 0.74,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.642,
     "centroid_med": 1.978,
     "hd95_med": 4.743,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.722,
     "centroid_med": 1.887,
     "hd95_med": 3.0,
     "n": 43
    }
   },
   "flag": null
  },
  "B5_unigradicon__baseline": {
   "condition": "B5_unigradicon__baseline",
   "group": "baselines",
   "status": "ok",
   "n_studies": 8,
   "n_studies_total_available": 8,
   "n_studies_matched_available": 8,
   "overall": {
    "dice_med": 0.818,
    "dice_q1": 0.646,
    "dice_q3": 0.91,
    "centroid_med": 1.205,
    "hd95_med": 2.598,
    "n_rows": 176
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.938,
     "centroid_med": 1.247,
     "hd95_med": 3.798,
     "n": 16
    },
    "spleen": {
     "dice_med": 0.918,
     "centroid_med": 1.155,
     "hd95_med": 2.121,
     "n": 16
    },
    "kidney_right": {
     "dice_med": 0.715,
     "centroid_med": 1.351,
     "hd95_med": 3.0,
     "n": 16
    },
    "kidney_left": {
     "dice_med": 0.712,
     "centroid_med": 0.926,
     "hd95_med": 2.976,
     "n": 16
    },
    "aorta": {
     "dice_med": 0.605,
     "centroid_med": 7.728,
     "hd95_med": 3.798,
     "n": 16
    },
    "inferior_vena_cava": {
     "dice_med": 0.633,
     "centroid_med": 4.07,
     "hd95_med": 3.927,
     "n": 16
    },
    "vertebrae_L1": {
     "dice_med": 0.895,
     "centroid_med": 0.714,
     "hd95_med": 1.5,
     "n": 16
    },
    "vertebrae_L2": {
     "dice_med": 0.898,
     "centroid_med": 0.787,
     "hd95_med": 1.5,
     "n": 16
    },
    "vertebrae_L3": {
     "dice_med": 0.907,
     "centroid_med": 0.771,
     "hd95_med": 1.5,
     "n": 16
    },
    "pancreas": {
     "dice_med": 0.658,
     "centroid_med": 1.814,
     "hd95_med": 4.609,
     "n": 16
    },
    "gallbladder": {
     "dice_med": 0.763,
     "centroid_med": 1.715,
     "hd95_med": 2.598,
     "n": 16
    }
   },
   "flag": null
  },
  "B5_unigradicon__raw": {
   "condition": "B5_unigradicon__raw",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.738,
    "dice_q1": 0.479,
    "dice_q3": 0.904,
    "centroid_med": 1.807,
    "hd95_med": 3.354,
    "n_rows": 583
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.944,
     "centroid_med": 1.554,
     "hd95_med": 4.243,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.919,
     "centroid_med": 1.424,
     "hd95_med": 2.598,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.628,
     "centroid_med": 1.219,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.686,
     "centroid_med": 1.671,
     "hd95_med": 3.354,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.558,
     "centroid_med": 8.882,
     "hd95_med": 4.5,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.527,
     "centroid_med": 7.711,
     "hd95_med": 5.196,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.881,
     "centroid_med": 0.822,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.887,
     "centroid_med": 0.756,
     "hd95_med": 1.5,
     "n": 51
    },
    "vertebrae_L3": {
     "dice_med": 0.908,
     "centroid_med": 0.688,
     "hd95_med": 1.5,
     "n": 49
    },
    "pancreas": {
     "dice_med": 0.612,
     "centroid_med": 2.253,
     "hd95_med": 5.408,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.672,
     "centroid_med": 3.061,
     "hd95_med": 3.354,
     "n": 43
    }
   },
   "flag": null
  },
  "B5_unigradicon_io__aligned": {
   "condition": "B5_unigradicon_io__aligned",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 26,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.832,
    "dice_q1": 0.667,
    "dice_q3": 0.924,
    "centroid_med": 1.148,
    "hd95_med": 2.598,
    "n_rows": 578
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.956,
     "centroid_med": 1.032,
     "hd95_med": 3.0,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.931,
     "centroid_med": 0.913,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.745,
     "centroid_med": 0.913,
     "hd95_med": 3.0,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.78,
     "centroid_med": 0.821,
     "hd95_med": 2.598,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.691,
     "centroid_med": 5.081,
     "hd95_med": 3.674,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.651,
     "centroid_med": 6.045,
     "hd95_med": 4.243,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.902,
     "centroid_med": 0.718,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.916,
     "centroid_med": 0.653,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.927,
     "centroid_med": 0.639,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.666,
     "centroid_med": 2.598,
     "hd95_med": 4.743,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.728,
     "centroid_med": 1.994,
     "hd95_med": 3.0,
     "n": 43
    }
   },
   "flag": null
  },
  "B5_unigradicon_io__baseline": {
   "condition": "B5_unigradicon_io__baseline",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.835,
    "dice_q1": 0.672,
    "dice_q3": 0.925,
    "centroid_med": 1.113,
    "hd95_med": 2.598,
    "n_rows": 578
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.956,
     "centroid_med": 1.03,
     "hd95_med": 3.0,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.932,
     "centroid_med": 0.908,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.741,
     "centroid_med": 0.937,
     "hd95_med": 2.598,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.779,
     "centroid_med": 0.782,
     "hd95_med": 2.598,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.689,
     "centroid_med": 5.677,
     "hd95_med": 3.674,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.654,
     "centroid_med": 6.386,
     "hd95_med": 3.674,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.902,
     "centroid_med": 0.718,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.917,
     "centroid_med": 0.634,
     "hd95_med": 1.5,
     "n": 49
    },
    "vertebrae_L3": {
     "dice_med": 0.929,
     "centroid_med": 0.564,
     "hd95_med": 1.5,
     "n": 46
    },
    "pancreas": {
     "dice_med": 0.666,
     "centroid_med": 2.364,
     "hd95_med": 4.743,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.73,
     "centroid_med": 1.986,
     "hd95_med": 3.0,
     "n": 43
    }
   },
   "flag": null
  },
  "B5_unigradicon_io__raw": {
   "condition": "B5_unigradicon_io__raw",
   "group": "baselines",
   "status": "ok",
   "n_studies": 26,
   "n_studies_total_available": 29,
   "n_studies_matched_available": 26,
   "overall": {
    "dice_med": 0.799,
    "dice_q1": 0.596,
    "dice_q3": 0.92,
    "centroid_med": 1.239,
    "hd95_med": 3.0,
    "n_rows": 583
   },
   "per_organ": {
    "liver": {
     "dice_med": 0.953,
     "centroid_med": 1.048,
     "hd95_med": 3.354,
     "n": 55
    },
    "spleen": {
     "dice_med": 0.928,
     "centroid_med": 0.935,
     "hd95_med": 2.121,
     "n": 55
    },
    "kidney_right": {
     "dice_med": 0.731,
     "centroid_med": 0.711,
     "hd95_med": 3.354,
     "n": 55
    },
    "kidney_left": {
     "dice_med": 0.779,
     "centroid_med": 0.591,
     "hd95_med": 2.598,
     "n": 55
    },
    "aorta": {
     "dice_med": 0.64,
     "centroid_med": 9.78,
     "hd95_med": 4.5,
     "n": 55
    },
    "inferior_vena_cava": {
     "dice_med": 0.615,
     "centroid_med": 7.417,
     "hd95_med": 4.743,
     "n": 55
    },
    "vertebrae_L1": {
     "dice_med": 0.908,
     "centroid_med": 0.548,
     "hd95_med": 1.5,
     "n": 55
    },
    "vertebrae_L2": {
     "dice_med": 0.92,
     "centroid_med": 0.45,
     "hd95_med": 1.5,
     "n": 51
    },
    "vertebrae_L3": {
     "dice_med": 0.92,
     "centroid_med": 0.451,
     "hd95_med": 1.5,
     "n": 49
    },
    "pancreas": {
     "dice_med": 0.65,
     "centroid_med": 2.444,
     "hd95_med": 5.185,
     "n": 55
    },
    "gallbladder": {
     "dice_med": 0.65,
     "centroid_med": 2.342,
     "hd95_med": 3.354,
     "n": 43
    }
   },
   "flag": null
  }
 }
};

const GROUP_COLORS = {
  ablation_aligned: "#0E7C86",
  ablation_noalign: "#5B4B8A",
  r_series: "#B8862B",
  baselines: "#3B5D72",
};

const METRICS = {
  dice: { label: "Eroded Dice", unit: "", better: "high", decimals: 2, field: "dice_med" },
  centroid: { label: "Centroid", unit: "mm", better: "low", decimals: 1, field: "centroid_med" },
  hd95: { label: "HD95", unit: "mm", better: "low", decimals: 1, field: "hd95_med" },
};

function fmt(x, decimals) {
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2013";
  return x.toFixed(decimals);
}

// good (green) -> mid (straw) -> bad (red) diverging scale
function colorForNorm(norm) {
  // norm: 0 = worst, 1 = best
  const clamped = Math.max(0, Math.min(1, norm));
  let r, g, b;
  if (clamped < 0.5) {
    // red (#c0463a) -> straw (#e9c768)
    const t = clamped / 0.5;
    r = Math.round(192 + (233 - 192) * t);
    g = Math.round(70 + (199 - 70) * t);
    b = Math.round(58 + (104 - 58) * t);
  } else {
    // straw (#e9c768) -> teal-green (#1e8f6f)
    const t = (clamped - 0.5) / 0.5;
    r = Math.round(233 + (30 - 233) * t);
    g = Math.round(199 + (143 - 199) * t);
    b = Math.round(104 + (111 - 104) * t);
  }
  return `rgb(${r},${g},${b})`;
}

function textColorForNorm(norm) {
  return norm > 0.18 && norm < 0.82 ? "#1c2126" : "#ffffff";
}

export default function OrganHeatmapDashboard() {
  const [metric, setMetric] = useState("dice");
  const [hideMissing, setHideMissing] = useState(false);
  const metricDef = METRICS[metric];

  const groupKeys = Object.keys(DATA.groups);
  const organs = DATA.organs;
  const columns = [...organs, "OVERALL"];

  function cellValue(rec, col) {
    if (!rec.overall) return null;
    if (col === "OVERALL") return rec.overall[metricDef.field];
    const o = rec.per_organ[col];
    return o ? o[metricDef.field] : null;
  }

  const range = useMemo(() => {
    let vals = [];
    Object.values(DATA.conditions).forEach((rec) => {
      columns.forEach((col) => {
        const v = cellValue(rec, col);
        if (v !== null && v !== undefined) vals.push(v);
      });
    });
    if (vals.length === 0) return { min: 0, max: 1 };
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, [metric]);

  function normFor(v) {
    if (v === null || v === undefined) return null;
    const span = range.max - range.min || 1;
    const raw = (v - range.min) / span;
    return metricDef.better === "high" ? raw : 1 - raw;
  }

  const legendStops = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="hm-root">
      <style>{`
        .hm-root {
          --ink: #10151c;
          --ink-soft: #4a5562;
          --paper: #f4f6f8;
          --panel: #ffffff;
          --line: #dde3e8;
          --teal: #0E7C86;
          --amber: #b8862b;
          --red: #b23a2e;
          --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
          --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
          font-family: var(--sans);
          background: var(--paper);
          color: var(--ink);
          padding: 28px;
          min-height: 100%;
          box-sizing: border-box;
        }
        .hm-root * { box-sizing: border-box; }
        .hm-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 24px;
          margin-bottom: 16px;
          flex-wrap: wrap;
        }
        .hm-title { font-size: 19px; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 4px 0; }
        .hm-sub { font-size: 12.5px; color: var(--ink-soft); margin: 0; max-width: 520px; line-height: 1.5; }
        .hm-controls {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 14px;
          flex-wrap: wrap;
        }
        .hm-metric-toggle {
          display: flex;
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 3px;
          gap: 2px;
        }
        .hm-metric-btn {
          font-family: var(--mono);
          font-size: 11.5px;
          font-weight: 600;
          padding: 6px 14px;
          border-radius: 6px;
          border: none;
          background: transparent;
          color: var(--ink-soft);
          cursor: pointer;
          text-transform: uppercase;
          letter-spacing: 0.02em;
        }
        .hm-metric-btn.active { background: var(--teal); color: white; }
        .hm-right { display: flex; align-items: center; gap: 18px; }
        .hm-checkbox {
          display: flex; align-items: center; gap: 6px;
          font-size: 12px; color: var(--ink-soft); font-weight: 600;
          cursor: pointer; user-select: none;
        }
        .hm-checkbox input { accent-color: var(--teal); width: 14px; height: 14px; cursor: pointer; }
        .hm-legend { display: flex; align-items: center; gap: 6px; font-size: 10.5px; color: var(--ink-soft); font-family: var(--mono); }
        .hm-legend .swatch-strip { display: flex; height: 10px; width: 90px; border-radius: 3px; overflow: hidden; border: 1px solid var(--line); }
        .hm-legend .swatch-strip span { flex: 1; height: 100%; }
        .hm-table-wrap {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 10px;
          overflow: auto;
          max-height: 640px;
          position: relative;
        }
        table.hm-table { border-collapse: collapse; font-size: 12px; width: 100%; }
        .hm-table thead th {
          position: sticky; top: 0; z-index: 2;
          background: #fafbfc;
          text-align: center;
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: 0.03em;
          color: var(--ink-soft);
          font-weight: 700;
          padding: 9px 8px;
          border-bottom: 1px solid var(--line);
          white-space: nowrap;
        }
        .hm-table thead th.sticky-col { left: 0; z-index: 3; text-align: left; }
        .hm-table thead th.overall-col { background: #f1f4f5; border-left: 2px solid var(--line); }
        .hm-table tbody tr.group-row td {
          padding: 7px 12px 3px 12px;
          font-size: 10px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--ink-soft);
          background: #fafbfc;
          position: sticky;
          left: 0;
        }
        .hm-table tbody tr.group-row { background: #fafbfc; }
        td.sticky-col, th.sticky-col {
          position: sticky;
          left: 0;
          background: var(--panel);
          z-index: 1;
          border-right: 1px solid var(--line);
        }
        .hm-table tbody tr:hover td.sticky-col { background: #f7faf9; }
        td.cond-cell {
          padding: 7px 12px;
          font-weight: 600;
          font-size: 11.5px;
          color: var(--ink);
          white-space: nowrap;
          min-width: 230px;
        }
        td.cond-cell.dim { color: #9aa4ad; font-weight: 500; }
        .cond-row-inner { display: flex; align-items: center; gap: 8px; }
        .group-bar { width: 3px; height: 15px; border-radius: 2px; flex-shrink: 0; }
        .mini-badge {
          font-size: 8.5px; font-weight: 800; letter-spacing: 0.03em;
          padding: 1px 5px; border-radius: 8px; flex-shrink: 0;
        }
        .mini-badge.partial { background: #fbf1e1; color: #92600f; }
        .mini-badge.missing { background: #f7e9e7; color: #8a2e22; }
        .flag-icon { font-size: 10px; color: var(--amber); cursor: help; flex-shrink: 0; }
        td.metric-cell {
          text-align: center;
          font-family: var(--mono);
          font-weight: 600;
          font-size: 11.5px;
          padding: 7px 6px;
          min-width: 58px;
          border-right: 1px solid rgba(255,255,255,0.5);
        }
        td.metric-cell.empty { color: #b7bfc5; background: #f6f7f8 !important; font-weight: 400; }
        td.metric-cell.overall-col { border-left: 2px solid var(--line); font-weight: 800; }
        tr.data-row.row-missing td.metric-cell { opacity: 0.35; }
      `}</style>

      <div className="hm-head">
        <div>
          <h1 className="hm-title">Registration Pipeline &mdash; Per-Organ Heatmap</h1>
          <p className="hm-sub">
            Median value per condition &times; organ, computed on each condition's matched or
            fallback study subset (see status dashboard). Cells colored green (better) to red
            (worse) within the current metric's full range.
          </p>
        </div>
        <div className="hm-legend">
          worst
          <span className="swatch-strip">
            {legendStops.map((s, i) => (
              <span key={i} style={{ background: colorForNorm(s) }} />
            ))}
          </span>
          best
        </div>
      </div>

      <div className="hm-controls">
        <div className="hm-metric-toggle">
          {Object.entries(METRICS).map(([k, m]) => (
            <button
              key={k}
              className={"hm-metric-btn" + (metric === k ? " active" : "")}
              onClick={() => setMetric(k)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="hm-right">
          <label className="hm-checkbox">
            <input
              type="checkbox"
              checked={hideMissing}
              onChange={(e) => setHideMissing(e.target.checked)}
            />
            Hide missing
          </label>
        </div>
      </div>

      <div className="hm-table-wrap">
        <table className="hm-table">
          <thead>
            <tr>
              <th className="sticky-col">Condition</th>
              {organs.map((o) => (
                <th key={o}>{o.replace(/_/g, " ")}</th>
              ))}
              <th className="overall-col">Overall</th>
            </tr>
          </thead>
          <tbody>
            {groupKeys.map((g) => {
              const conds = DATA.groups[g].conditions
                .map((c) => DATA.conditions[c])
                .filter((rec) => !(hideMissing && rec.status === "missing"));
              if (conds.length === 0) return null;
              return (
                <React.Fragment key={g}>
                  <tr className="group-row">
                    <td
                      colSpan={columns.length + 1}
                      style={{ borderLeft: `3px solid ${GROUP_COLORS[g]}` }}
                    >
                      {DATA.groups[g].label}
                    </td>
                  </tr>
                  {conds.map((rec) => {
                    const isMissing = rec.status === "missing";
                    return (
                      <tr key={rec.condition} className={"data-row" + (isMissing ? " row-missing" : "")}>
                        <td className={"sticky-col cond-cell" + (isMissing ? " dim" : "")}>
                          <div className="cond-row-inner">
                            <span className="group-bar" style={{ background: GROUP_COLORS[g] }} />
                            {rec.condition}
                            {rec.flag && <span className="flag-icon" title={rec.flag}>&#9888;</span>}
                            {rec.status === "partial" && <span className="mini-badge partial">PARTIAL</span>}
                            {rec.status === "missing" && <span className="mini-badge missing">MISSING</span>}
                          </div>
                        </td>
                        {columns.map((col) => {
                          const v = cellValue(rec, col);
                          const norm = normFor(v);
                          if (v === null || v === undefined) {
                            return (
                              <td
                                key={col}
                                className={"metric-cell empty" + (col === "OVERALL" ? " overall-col" : "")}
                              >
                                &#8211;
                              </td>
                            );
                          }
                          return (
                            <td
                              key={col}
                              className={"metric-cell" + (col === "OVERALL" ? " overall-col" : "")}
                              style={{
                                background: colorForNorm(norm),
                                color: textColorForNorm(norm),
                              }}
                            >
                              {fmt(v, metricDef.decimals)}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
