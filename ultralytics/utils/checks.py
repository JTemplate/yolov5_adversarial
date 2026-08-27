"""Small subset of ultralytics.utils.checks used by this repository."""


def check_requirements(requirements=(), *args, **kwargs):
    """Compatibility no-op; dependencies are managed by the active environment."""
    return True
