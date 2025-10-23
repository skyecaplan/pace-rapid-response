# %% Import and load modules
# --- Standard library imports ---
import os            # Operating system utilities
import sys           # System-specific parameters and functions


# --- Update the path for custom tools ---
util_path = os.path.expanduser(
'~/Documents/GitHub/pace-rapid-response/PRR_OC/Bloom_Detection/utilities'
)
sys.path.append(util_path)

# --- Custom tool imports (reload for development convenience) ---
# import detection_util_MK
# importlib.reload(detection_util_MK)
from detection_util_MK import Bloom_Detection

# Run Code
Bloom_Detection('20250916')













# %%
