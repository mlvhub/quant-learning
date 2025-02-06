# Quant Learning

Various resources for learning quantitative finance, trading, and machine learning.

## Steps to create a new Python project (using uv)

1. Initialise a new project with `uv init $PROJECT_NAME`
2. `cd $PROJECT_NAME` into the project folder
3. Activate the virtual environment with `source .venv/bin/activate`
4. Install the necessary dependencies, e.g.:
```bash
uv add jupyter pandas matplotlib numpy yfinance polars python-dotenv
```

### Old with Pipenv

1. Create a new folder for the project and `cd` into it
2. Create a new virtual environment:
```bash
pipenv install jupyter pandas polars python-dotenv matplotlib numpy yfinance
```
3. Run `pipenv shell` to activate the virtual environment
4. Create a new `.env` file and add the necessary variables, if necessary
5. Run `jupyter notebook --no-browser` to start the Jupyter notebook server
