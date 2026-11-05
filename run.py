"""Reproduce the maintained unanimous-expert prediction pipeline.

Example:
    python run.py --data /path/to/official-data \
        --output submissions/submission.csv
"""

from pokemon.consensus import main


if __name__ == "__main__":
    main()
