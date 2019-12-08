# Cross-Document Event Coreference (CDEC)

Code to perform CDEC, which is the task of clustering event mentions in a collection
of documents that refer to the same real-world event. Configured to do CDEC on the ECB+ corpus using only event trigger annotations. Matches or slightly beats state of the art on trigger-only CDEC with a significantly simpler model.

## What is this for?
Suppose some event takes place and journalists
inform the public. The reports refer to the same
event but inevitably vary in their language, tone
and contextualization. What can we learn from
the variations? Of course, we must first group together all reports of the same event before we can
realize any analysis. Generating these event groupings automatically is called Cross-Document Event Coreference (CDEC). It is an important component not only for this application, but also for tasks
such as information retrieval and question answering. CDEC allows us to augment the information
around an event mention in a single document with
information from all of its mentions across many
documents, such as entities and other contextual
information, allowing us to paint a clearer picture
of how events and their encompassing stories are
told.

## Prerequisites
**Note:** This code was developed on Ubuntu 19.10, and instructions assume access to a linux shell.

### Data
[ECB+](http://www.newsreader-project.eu/results/data/the-ecb-corpus/)
### Dependencies
- Download pre-trained Fasttext vectors [here](https://fasttext.cc/docs/en/crawl-vectors.html) -- select the English bin file.
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html) 3.9.0
- The official [CoNLL Scorer](https://github.com/conll/reference-coreference-scorers)
- ``java 1.8``
- ``python 3.7.5``
- ``pipenv 2018.11.26``
- ``maven 3.6.1``
- Dependencies for python files ('python assets') are listed in corresponding Pipfile and managed through ``pipenv``.
- Dependencies for main java project ('cdec') are in pom.xml and managed through ``maven``.

## Instructions
1. Clone the CoNLL scorer repo and place it into a directory called "perl_assets" beneath the root directory.
  ```bash
  mkdir perl_assets
  cd perl_assets
  git clone https://github.com/conll/reference-coreference-scorers
  ```
2. Open the root directory in a terminal and type the following commands to generate the ``pipenv`` environments for the required python scripts:

  ```bash
  cd python_assets
  cd ecb_augmenter
  pipenv install
  cd ../word_vecs
  pipenv install
  ```
3. Prepare the ECB+ corpus:

  - First, open your download of the ECB+ corpus and place the extracted ECB+.zip directory and the file "ECBplus_coreference_sentences.csv" in the ``CDEC/data`` directory.
  - "Augment" the ECB+ corpus (this makes it more convenient to parse):
    ```bash
    cd ecb_augmenter
    pipenv run python main.py
    ```
4. Open the file "external_paths.json" in the root directory and add the paths to your installation of Stanford CoreNLP and the Fasstext vectors.

5. Start the word embedding server:
  ```bash
  cd word_vecs
  pipenv run python main.py
  ```
  When the server is finished loading your terminal will display a message with the server's port.

6. Run the main java project:
  ```bash
  cd cdec
  java -cp target/CDEC-0.0.1-SNAPSHOT-jar-with-dependencies.jar main.Main
  ```
  This will run random 5-fold cross validation and log the results to an automatically created folder at the same level of the root directory ``data/results``.

  If you wish to compile the code from source, use maven to package the pom.xml file.

7. Remember to exit the CoreNLP server and the word embedding server once you are done.
