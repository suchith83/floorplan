import warnings

# numpy + Apple Accelerate raise spurious "divide by zero / overflow / invalid value encountered in
# matmul" warnings on Apple silicon; results are finite (checked on all three ARKitScenes scans).
warnings.filterwarnings("ignore", message=r".* encountered in matmul", category=RuntimeWarning)
