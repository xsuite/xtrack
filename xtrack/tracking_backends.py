"""Registry mapping a non-native particle type to its tracking backend.

Core xtrack never imports the backends (e.g. ``xtrack.tpsa``); the backend package
registers itself at import time.  ``BeamElement.track`` / ``Line.track`` consult
``backend_for`` only for objects that are not ``xt.Particles`` (the doubles path
is never touched).
"""

_BACKENDS = {}  # particle type -> backend instance


def register_tracking_backend(particle_type, backend):
    """Register ``backend`` for exactly ``particle_type``; subclasses register their own."""
    _BACKENDS[particle_type] = backend


def backend_for(particles):
    return _BACKENDS.get(type(particles))
