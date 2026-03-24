# Final File Map

This package normalizes the working files into stable names so you do not have to keep track of version suffixes.

## Stable final names

- `advanced_balanced_hybrid_persona_pipeline.py`  
  Final balanced hybrid generation pipeline.

- `pragmatist_hybrid_persona_integration.py`  
  Stable final integration bridge. This is the packaged version of the earlier `pragmatist_hybrid_persona_integration_v3.py`.

- `pragmatist_hybrid_persona_notebook_cells.py`  
  Stable final notebook cell helper. This is the packaged version of the earlier `pragmatist_hybrid_persona_notebook_cells_v3.py`.

- `pragmatist_hybrid_context_error_engineering.py`  
  Stable final context-error engineering layer.

## Why the names were normalized

The earlier artifacts were created in stages. Version suffixes were useful during iteration, but they make the final handoff harder to follow. The package therefore keeps the final working code while removing the suffixes from the filenames that you should use going forward.

