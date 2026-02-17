# %% Import and load modules
# --- Standard library imports ---
import os            # Operating system utilities
import sys           # System-specific parameters and functions
import shutil        # High-level file operations


# --- Update the path for custom tools ---
util_path = os.path.expanduser(
'~/Documents/GitHub/pace-rapid-response/PRR_OC/Bloom_Detection/utilities'
)
sys.path.append(util_path)

# --- Custom tool imports (reload for development convenience) ---
# import detection_util_MK
# importlib.reload(detection_util_MK)
from detection_util_MK import Bloom_Detection
from gt_html_utils_MK import write_full_html

# %% Single Configuration Single Day Run Example
day = '20260202'
days = 15
anomaly_type = 'relative'
anomaly_threshold = 100
base_dir = os.path.dirname(__file__)
out_path = os.path.join(base_dir, 'figures', day, 'html', f'OCI_chlor_a_anomaly_daily_{day}.html')
move_path = os.path.join(base_dir, 'figures', day)

Bloom_Detection(day, anomaly_type=anomaly_type, anomaly_threshold=anomaly_threshold, days_prior=days, dpi=200, delete_flag=False)
write_full_html(day, out_path)


# %% Multi-configuration Single Day Run
# day = '20260202'
# base_dir = os.path.dirname(__file__)
# out_path = os.path.join(base_dir, 'figures', day, 'html', f'OCI_chlor_a_anomaly_daily_{day}.html')
# move_path = os.path.join(base_dir, 'figures', day)

# days = [15, 30, 60]
# anomaly_type = 'absolute'
# threshold = [0.5, 1]

# # Run Bloom Detection and generate HTML for each absolute anomaly configuration
# for days_prior in days:
#     for anomaly_threshold in threshold:
#         Bloom_Detection(day, anomaly_type=anomaly_type, anomaly_threshold=anomaly_threshold, days_prior=days_prior, dpi=200, delete_flag=False)
#         write_full_html(day, out_path)

#         # Move generated figures/html to appropriate directory (if needed)
#         # Create new figure directory using day_days_prior_anomaly_threshold_100x100_10p_5000val format
#         new_dir_name = f"{day}_{days_prior}_{anomaly_threshold}_100x100_10p_5000val"
#         new_dir_path = os.path.join(base_dir, 'figures', new_dir_name)

#         # Remove destination if it exists
#         if os.path.exists(new_dir_path):
#             print(f"Directory {new_dir_path} already exists. Removing it.")
#             shutil.rmtree(new_dir_path)

#         # Move the entire directory (renames it)
#         print(f"Moving directory from {move_path} to {new_dir_path}")
#         shutil.move(move_path, new_dir_path)

# anomaly_type = 'relative'
# threshold = [50, 75, 100, 150]

# # Run Bloom Detection and generate HTML for each relative anomaly configuration
# for days_prior in days:
#     for anomaly_threshold in threshold:
#         Bloom_Detection(day, anomaly_type=anomaly_type, anomaly_threshold=anomaly_threshold, days_prior=days_prior, dpi=200, delete_flag=False)
#         write_full_html(day, out_path)

#         # Move generated figures/html to appropriate directory (if needed)
#         # Create new figure directory using day_days_prior_anomaly_threshold_100x100_10p_5000val format
#         new_dir_name = f"{day}_{days_prior}_{anomaly_threshold}pct_100x100_10p_5000val"
#         new_dir_path = os.path.join(base_dir, 'figures', new_dir_name)

#         # Remove destination if it exists
#         if os.path.exists(new_dir_path):
#             print(f"Directory {new_dir_path} already exists. Removing it.")
#             shutil.rmtree(new_dir_path)

#         # Move the entire directory (renames it)
#         print(f"Moving directory from {move_path} to {new_dir_path}")
#         shutil.move(move_path, new_dir_path)


