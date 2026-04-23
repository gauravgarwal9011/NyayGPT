# __init__.py marks this directory as a Python "package".
# A package is a folder Python treats as importable, like a namespace.
# Without this file, you couldn't write `from kd_pipeline.config import ...`.
#
# We expose the version string so other modules / scripts can introspect it
# (e.g., for logging, debugging, or printing in the CLI banner).
__version__ = "1.0.0"
