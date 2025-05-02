# Football Scouting RAG Pipeline

## Project Goal

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to simulate a professional football scouting process. It takes raw text scouting reports as input, allows users to query these reports with natural language questions (e.g., "Find me a fast winger with good dribbling skills"), and uses a Large Language Model (llmama3.3 70b) acting as an expert scout to generate detailed answers and also provide ranking of scouted players.


## Project Structure

```
Project/
│
├── football_scouting/
│   ├── data/
│   │   └── players_review.txt  # Example (or other .txt scouting reports)
│   ├── .env                # API Keys 
│   ├── main.py             # Main script to run the RAG pipeline & interact with user
│   ├── utils.py            # Utility functions (API key loading, model init, data loading/chunking)
│   └── requirements.txt    # Project dependencies
│
└── README.md               # This file
```

## Setup Instructions

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-url>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Navigate to the `football_scouting` directory and install the required packages:
    ```bash
    cd football_scouting
    pip install -r requirements.txt
    cd .. 
    ```

4.  **Set Up API Key:**
    *   Create a file named `.env` inside the `football_scouting` directory.
    *   Add your Groq API key to the file:
        ```env
        GROQ_API_KEY=your_groq_api_key_here
        ```
    * 
5.  **Add Scouting Reports:**
    *   Place your scouting reports as individual `.txt` files inside the `football_scouting/data/` directory. The script `utils.py` will automatically load all `.txt` files from this folder. (Sample synthetic data is included in `players_review.txt`).

## How to Run

1.  Ensure your virtual environment is activated and you are in the **root directory**.
2.  Run the main script:
    ```bash
    python football_scouting/main.py
    ```
3.  The script will:
    *   Load the API key.
    *   Initialize the embedding model (may take time on first run as the model downloads).
    *   Initialize the Groq LLM.
    *   Load and chunk the scouting reports from `football_scouting/data/`.
    *   Build the FAISS vector store.
    *   Prompt you to enter your query (e.g., "Find a defender who is good in the air").
4.  Enter your queries when prompted. Type `quit` to exit the application.

