# Preformance Scorer for your project

This package contains a scorer for your project. It is designed to be run from the command line, and will test your project against a simple set of metrics
so that we (and you) can see your progress.

You should run this from the command line, with the following command:

```bash
python -m scoring
```

This will run the scorer, and print out a report of your project's performance (run the base line immediately to see what the defaults
look like).

If you run pytest on the scoring module, it will perform some basic tests
on your package to check that we can run the scorer on it.