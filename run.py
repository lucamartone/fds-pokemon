"""Run the maintained candidate 7010 using the official competition files.

Example:
    python run.py --data /path/to/official-data \
        --output submissions/submission_candidate_7010.csv
"""

from pokemon.pipeline import main


if __name__ == "__main__":
    main()
