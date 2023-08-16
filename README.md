# crank-that-pydocs

Scrape website text to a faiss db and query that db with a chatbot.

To dos:
1. Make pydocs-query return meaningful text, not vector addresses.

## Run the program

1. Run `pydocs-load-db.py` to vectorize the content at the URL on line 13.
A /faiss directory will be created.

2. Run `pydocs-query.py` to query the db at faiss/index.faiss.
The query currently only returns the vector addresses, not meaningful text.


## Virtual env for testing
Create virtual environment and install packages with pip3.
If you use "test" for your venv-name, git won't track it.

```python3
python3 -m venv <venv-name>
source <venv-name>/bin/activate
pip3 install langchain sentence_transformers faiss-cpu ctransformers
```

To freeze current dependencies into a requirements.txt file:

```python3
python3 -m pip freeze > requirements.txt
```

To build project with requirements.txt:
```python3
python3 -m pip install -r requirements.txt
```

Type "deactivate" to leave venv.