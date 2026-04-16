from phasmix.spiral import generate_multi_bin_spirals_bt21
import os

try:
    print("Testing generate_multi_bin_spirals_bt21 with n_samples=1000...")
    generate_multi_bin_spirals_bt21(output_file="test_spiral.png", n_samples=1000, r_bins=[(7.0, 7.5)])
    print("Test successful!")
    if os.path.exists("test_spiral.png"):
        print("Output file test_spiral.png created.")
        os.remove("test_spiral.png")
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
