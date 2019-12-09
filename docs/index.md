# Cross-Document Event Coreference (CDEC)

This project was completed as the Capstone project for the M.S. in Data Science at Florida International University. The system perfoms "Cross-Document Event Coreference" (described below), achieving state-of-the-art performance amongst trigger-only systems with a significantly simpler model than prior work. The code is available, with running instructions, in the github repo linked above.

## What is CDEC for?
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
