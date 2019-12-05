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
### Data
[ECB+](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).
### External Software
- Download pre-trained Fasttext vectors [here](https://fasttext.cc/docs/en/crawl-vectors.html) -- select the English bin file.
- [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html) 3.8.0

Further dependencies for python files are listed in corresponding Pipfile. Java dependencies are in the pom.xml file.

## Instructions
... in progress
