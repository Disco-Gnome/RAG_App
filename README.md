# Custom RAG Engine

This is a GenAI Streamlit application for RAG, allowing users to chat with agents using their own custom library of .pdfs.

## Features
- Process .pdf files using different approaches (selected by user)
- Easy setup and use

## Installation and Setup

### Prerequisites

This application requires an installation of Python 3.12 or later.

### Local installation

1. Clone this repository to your local machine.
   
2. Navigate to the RAG_App directory.

   ```bash
   cd RAG_App
   ```
   
3. Install the required dependencies using pip in a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```
   
4. In the root directory, add your API keys to an `.env` file.
   
5. Open the `.env` file (create one if not present) using any text editor and enter your API keys as shown below:

   ```bash
   export GOOGLE_GEMINI_API_KEY = "YOUR GOOGLE GEMINI KEY"
   export ANTHROPIC_API_KEY = "YOUR ANTROPIC API KEY"
   export OPENAI_API_KEY = "YOUR OPENAI API KEY"
   ```
   If using Azure for OPEN AI, also include
   ```bash
   export OPENAI_API_BASE = "YOUR OPENAI API BASE"
   export OPENAI_API_TYPE = "YOUR OPENAI API TYPE"
   export OPENAI_API_VERSION = "YOUR OPENAI API VERSION"
   ```

## Running the application

To run the application locally:

```bash
cd src
streamlit run app.py
```

To run in a container:
```bash
docker build -t RAG_App .
docker run -p 8501:8501 RAG_App
```
And open the app at http://localhost:8501/

## License

This project is licensed under the terms of the Apache License 2.0. Please see the [LICENSE](LICENSE) file for details.

## Attribution

This project is built upon and modifies PDFLucy. This repository contains modifications and additional functionality developed by Ryan McNeil.

## Support

For any questions or issues, please contact the maintainers, or raise an issue in the GitHub repository.
