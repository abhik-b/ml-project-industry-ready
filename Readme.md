## Steps

1. Created a Github repository.
2. Opened conda prompt & then opened a folder in vs code from conda prompt.
3. Created a conda environment and activated it

```
conda create -p venv python==3.8 -y
conda activate venv/
```

4. Then created `setup.py` & `pyproject.toml`. These files will help us in developing our application as a package.
5. Then created `src` folder. The setup.py will build whatever code is there inside this src folder as the package.
