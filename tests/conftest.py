import os

# Force test-safe API key BEFORE any test module imports backend.raphi_server.
# The production RAPHI_API_KEY from the shell environment must not bleed into
# the test process — the TokenAuth middleware captures api_key at import time.
os.environ["RAPHI_API_KEY"] = "test-key"
