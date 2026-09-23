"""Notebook helpers: transcript parsing (board_logs), Table 1 (exp1), the demo signal (demo), tables (report)."""
import warnings as _warnings

# Library notices that print absolute install paths into published notebook outputs and say nothing about
# the results: tqdm without ipywidgets, and music2latent disabling CUDA autocast on a CPU.
_warnings.filterwarnings("ignore", message="IProgress not found")
_warnings.filterwarnings("ignore", message="CUDA is not available")
