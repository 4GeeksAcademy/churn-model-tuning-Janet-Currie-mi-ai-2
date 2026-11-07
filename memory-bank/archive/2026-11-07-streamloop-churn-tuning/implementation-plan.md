# Implementation plan

1. Create the environment and pin the libraries.
2. Add the blank-charge cleaner and a small standard-library test.
3. Build the notebook in the required order: check, split, factory model, broad search, tight grid, stability choice, one final test.
4. Choose the tight grid only after the broad search showed the promising region.
5. Write `tuning_report.md` from the printed scores.
6. Update the README files and the course-card title and description.
7. Run the notebook again from a clean start and match the report.
