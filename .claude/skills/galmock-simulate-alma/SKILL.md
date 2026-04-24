---
name: galmock-simulate-alma
description: >
  Use this skill when the user asks to "simulate ALMA", "run simobserve", "create
  mock visibilities", "simulate an ALMA observation", "run Layer B", or "make a
  MeasurementSet".
---

# Layer B: Simulate ALMA Observation

Simulate ALMA interferometric observation of an intrinsic CO cube using
CASA `simobserve`, producing a MeasurementSet.

## Prerequisites

- **Environment**: CASA (`casatools`, `casatasks`).  Layer A must have been run
  first to produce `intrinsic_cube.fits`.
- The `galmockuv` package must be importable.

## Commands

```bash
# Run Layers B+C (simulation + measurement) in CASA
casa --nologger --nogui -c "
import sys, argparse, pathlib
sys.argv = ['galmockuv.py', 'config.toml', '--layers', 'B+C']
galmockuv_globals = {'__name__': '__main__', '__file__': 'galmockuv.py', 'sys': sys, 'argparse': argparse, 'pathlib': pathlib, '__builtins__': __builtins__}
exec(open('galmockuv.py').read(), galmockuv_globals)
"
```

Or programmatically (inside CASA):

```python
from galmockuv import run_pipeline
results = run_pipeline("config.toml", layers="B")
print(results["ms_path"])
```

## Output

| File | Description |
|------|-------------|
| `{source_id}.ms` | Simulated MeasurementSet (concatenated if multi-track) |
| `casa_sim_params.yaml` | CASA simulation parameters used |

## Config Parameters (ALMA simulation)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `antennalist` | `alma.cycle7.7` | ALMA array configuration |
| `totaltime` | `"600s"` | Total on-source time |
| `integration_s` | 10.0 | Correlator dump time |
| `thermalnoise` | `"tsys-atm"` | Noise model |
| `user_pwv` | 1.5 | Precipitable water vapor (mm) |
| `mapsize` | `"5arcsec"` | Field of view |
| `seed` | 42 | Random seed |

### Multi-track scheduling

For `totaltime` > `max_track_hours` (default 8), the integration is
automatically split into multiple short tracks centred on transit, then
concatenated.  This ensures image noise scales as 1/sqrt(t).  Override with:

```toml
max_track_hours = 8
```

## Key Functions

| Function | Description |
|----------|-------------|
| `simulate_alma(cube_path, config, output_dir)` | Simulate ALMA, return (ms_path, sim_params) |

## Troubleshooting

- **`exec(open(...).read())` fails in CASA**: Pass explicit globals dict (see Commands above).
- **Project directory not cleaned up**: CASA `simobserve` creates a project dir in CWD.
  The pipeline moves the MS to `output_dir` before cleanup.
- **Single-track MS deleted before weight fix**: Fixed — MS is now moved to `output_dir`
  before cleanup for both single and multi-track modes.
