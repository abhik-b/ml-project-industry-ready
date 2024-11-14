## Steps

1. Created a Github repository.
2. Opened a local folder with same name in vs code .
3. Created a virtual environment and activated it

```
virtualenv venv
.\venv\Scripts\activate
```

4. Then created `setup.py` & `pyproject.toml`. These files will help us in developing our application as a package.
5. Then created `src` folder. The setup.py will build whatever code is there inside this src folder as the package.
6. Then pip installed the `requirements.txt` using `pip install -r requirements.txt`
