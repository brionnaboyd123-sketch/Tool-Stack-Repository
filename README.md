# Tool-Stack-Repository
In order to provide experimental code for the associated paper.
\ Privacy Regulatory Validation Layer Code and scripts for our IEEE paper on automated regulatory compliance checking using local LLMs and legal ontologies. 

The pipeline runs 100% locally and offline—extracting rules from an OWL/RDF ontology, parsing statutory PDFs (like California SB 327), scoring compliance using deontic modality and keyword matching, and using a local Llama 3.2 model to generate an audit spreadsheet. 

Requirements -


Python 3.10+ - [Ollama](https://ollama.com/) (for running Llama 3.2 locally) Python libraries: '''bash pip install rdflib owlready2 pypdf pandas openpyxl spacy requests

Tool Stack required, 
RDFLib 

> Extracts raw RDF triples and annotations from ontology. 

Owlready2 

> Traverses class hierarchies and descendant lineages in graph. 

PyPDF 

> Parses statutory sentence clauses directly from local PDFs. 

Stanford Deontic Taxonomy (Regex) 

> Classifies legal statements into obligations, prohibitions, or permissions. 

Llama 3.2 (via Ollama) 

> Evaluates semantic legal alignment and generates qualitative reasoning. 

Pydantic 

> Enforces strict data schemas to eliminate JSON crashes. 

Pandas & Openpyxl 

> Exports multi-tab Excel reports for human audit verification.
