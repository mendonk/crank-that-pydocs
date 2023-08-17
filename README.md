# crank-that-pydocs

Scrape website text to a local faiss db and query that db with a chatbot.

To dos:
1. Add the model as a dependency instead of a manual download.
2. Error control and logging.
3. Fine-tune model for better results or replace altogether.
4. Make the chat stay open.
5. Add the option to scrape by sitemap.

## Download llama 2

This uses the lightest weight (2.7 GB) model of Llama 2 I could find.
7B isn't really built to be a chatbot, so the results aren't brilliant - it requires MORE POWER.
The only way I know to make this work right now is to download the model binary from Hugging Face [here](https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin) and put it into the root of this repository.

## Run the program

1. Run `pydocs-load-db.py` to vectorize the content at the URL on line 13.
Change the URL to query different websites.
A /faiss directory will be created.

2. Run `pydocs-query.py` to query the db at faiss/index.faiss.
The query will return an answer, sometimes with a bunch of text repeating.
Change the prompt on line 45 to ask different questions.

## Create virtual env for testing
Create virtual environment and install packages with pip3.
Use "test" for your venv-name so git won't track it.

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