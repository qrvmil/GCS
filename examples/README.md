# Examples

These examples require Python 3.12 and an editable installation of the project:

```bash
python -m pip install -e .
```

The first run may download robot models used by Drake and `manipulation`. The minimal
example runs one deterministic single-shelf query with random seed `42` and normally
finishes in a few seconds:

```bash
python examples/minimal_online.py
```

To run the same query with Meshcat visualization:

```bash
python examples/visualize_single_shelf.py
```

Open the Meshcat URL printed in the terminal. Stop the visualization process with
Ctrl+C when you are finished.
