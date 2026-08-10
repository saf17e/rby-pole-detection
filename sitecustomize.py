try:
    import torch

    lib = getattr(torch, "library", None)
    if lib is not None and not hasattr(lib, "register_fake"):
        def _register_fake(name):
            def _decorator(fn):
                # Best-effort no-op shim: return the function unchanged.
                return fn

            return _decorator

        setattr(lib, "register_fake", _register_fake)
except Exception:
    # If anything goes wrong here, let the original import path raise its error.
    pass
