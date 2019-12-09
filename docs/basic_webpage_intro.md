# Cross-Document Event Coreference (CDEC)

This project was done while the author was a graduate researcher at the Cognition, Narrative and Culture
lab at Florida International University. This system performs CDEC (explained below) on the ECB+
corpus, achieving state-of-the-art results among trigger-only systems, doing so with a significantly
simpler framework than prior work. The code and instructions to run it are in the github repo.

## Motivation
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
