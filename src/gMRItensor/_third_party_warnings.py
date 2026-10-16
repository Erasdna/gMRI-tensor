"""Warning filters for third-party packages, installed on package import."""
import warnings

# tensorly 0.9.0 has invalid escape sequences in docstrings (solvers/nnls.py,
# solvers/admm.py). Python warns each time it compiles them, which is every
# import (and every spawn worker) when site-packages is read-only, as in the
# Apptainer image. Compile-time warnings carry the file path as "module".
# Remove once tensorly ships fixed docstrings.
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r".*[/\\]tensorly[/\\]",
)
