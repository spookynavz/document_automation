# Document Automation with Instruction-Based Editing

This project automates the process of editing `.docx` documents using natural language instructions. It extracts instructions and content from separate files, semantically matches them using embeddings, and applies the modifications with the help of a language generation model like **LLaMA2** (via Ollama).

---

## Features

- Extracts instructions and content from `.docx` files
- Uses **SentenceTransformer** for semantic matching
- Applies instructions using LLMs like **LLaMA2** or **Falcon**
- Saves the updated content in a new `.docx` file
- Modular and easy to adapt to other models or instruction formats

---
## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) (if using local LLaMA2)
- Python libraries:

```bash
pip install sentence-transformers transformers python-docx nltk requests

---
<img width="491" height="224" alt="image" src="https://github.com/user-attachments/assets/45511bb6-0590-4122-b181-0b777d95ad01" />


