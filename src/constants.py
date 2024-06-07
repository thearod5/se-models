import os
from os.path import dirname

BASE_MODEL = "microsoft/codebert-base"
REPOSITORY_PATH = os.path.join(dirname(dirname(__file__)), "repositories")

# Repositories
PL_BERT_SINGLE_REPO_PATH = "thearod5/pl-bert-single"
NL_BERT_REPO_PATH = "thearod5/nl-bert"
