#    Copyright 2020 Division of Medical Image Computing, German Cancer
#    Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import shutil
import argparse
import logging
import multiprocessing
import psutil
from batchgenerators.utilities.file_and_folder_operations import *
import nnunet_mednext
from nnunet_mednext.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet_mednext.experiment_planning.utils import crop
from nnunet_mednext.paths import *
from nnunet_mednext.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet_mednext.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet_mednext.training.model_restore import recursive_find_python_class


# -------------------------------------------------------------------------
# SAFETY: safer multiprocessing start mode (prevents deadlocks)
# -------------------------------------------------------------------------
multiprocessing.set_start_method("spawn", force=True)

# -------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -------------------------------------------------------------------------
log_path = os.path.abspath("preprocess_log.txt")
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
print(f"[INFO] Logging to {log_path}")

def log(msg, level="info"):
    print(msg)
    getattr(logging, level)(msg)


# -------------------------------------------------------------------------
# DYNAMIC THREAD MANAGEMENT
# -------------------------------------------------------------------------
def adaptive_thread_limit(requested_threads: int, safety_factor: float = 0.75, est_gb_per_thread: float = 4.0):
    """
    Dynamically reduces threads if available RAM is too low.
    Prevents out-of-memory freezes during preprocessing.
    """
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    max_threads = int((avail_gb * safety_factor) / est_gb_per_thread)
    safe_threads = min(requested_threads, max(1, max_threads))
    if safe_threads < requested_threads:
        log(f"[INFO] Reducing threads from {requested_threads} → {safe_threads} "
            f"(available {avail_gb:.1f} GB RAM)")
    return safe_threads


# -------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of task IDs to preprocess")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21")
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D_v21")
    parser.add_argument("-no_pp", action="store_true", help="Skip preprocessing (planning only)")
    parser.add_argument("-tl", type=int, required=False, default=8, help="Threads for low-res preprocessing")
    parser.add_argument("-tf", type=int, required=False, default=8, help="Threads for full-res preprocessing")
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False)
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False)

    args = parser.parse_args()
    dont_run_preprocessing = args.no_pp

    # Apply adaptive thread limits
    tl = adaptive_thread_limit(args.tl)
    tf = adaptive_thread_limit(args.tf)

    planner_name3d = None if args.planner3d == "None" else args.planner3d
    planner_name2d = None if args.planner2d == "None" else args.planner2d

    if args.overwrite_plans is not None:
        if planner_name2d is not None:
            log("Overwriting plans only works for the 3D planner. Setting planner2d=None", "warning")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', \
            "When using --overwrite_plans you must use -pl3d ExperimentPlanner3D_v21_Pretrained"

    # Resolve task names
    tasks = []
    for i in args.task_ids:
        i = int(i)
        task_name = convert_id_to_task_name(i)

        if args.verify_dataset_integrity:
            log(f"Verifying dataset integrity for {task_name}")
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))

        crop(task_name, False, tf)
        tasks.append(task_name)

    # Load planners dynamically
    search_in = join(nnunet_mednext.__path__[0], "experiment_planning")
    planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet_mednext.experiment_planning") if planner_name3d else None
    planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet_mednext.experiment_planning") if planner_name2d else None

    # ---------------------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------------------
    for t in tasks:
        try:
            log(f"\n=== Starting Task: {t} ===")

            cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
            preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)

            # Skip already processed tasks
            if os.path.exists(join(preprocessing_output_dir_this_task, "nnUNetPlans.json")):
                log(f"[SKIP] {t} already preprocessed.", "warning")
                continue

            dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
            modalities = list(dataset_json["modality"].values())
            collect_intensityproperties = any(m.lower() == "ct" for m in modalities)

            log(f"Analyzing dataset {t} | modalities: {modalities}")
            dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)
            _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)

            maybe_mkdir_p(preprocessing_output_dir_this_task)
            shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
            shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

            threads = (tl, tf)
            log(f"Using threads: {threads}")

            # -----------------------------------------------------------------
            # Run 3D and/or 2D planners
            # -----------------------------------------------------------------
            if planner_3d is not None:
                if args.overwrite_plans is not None:
                    assert args.overwrite_plans_identifier is not None, "Need -overwrite_plans_identifier"
                    exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task,
                                             args.overwrite_plans, args.overwrite_plans_identifier)
                else:
                    exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
                exp_planner.plan_experiment()
                if not dont_run_preprocessing:
                    log(f"Running 3D preprocessing for {t}")
                    exp_planner.run_preprocessing(threads)

            if planner_2d is not None:
                exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
                exp_planner.plan_experiment()
                if not dont_run_preprocessing:
                    log(f"Running 2D preprocessing for {t}")
                    exp_planner.run_preprocessing(threads)

            log(f"✅ Completed preprocessing for {t}")

        except Exception as e:
            log(f"❌ ERROR in task {t}: {e}", "error")
            import traceback
            logging.error(traceback.format_exc())
            continue

    log("All requested tasks processed successfully.")


# -------------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
