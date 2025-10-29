# 🏛️ Museum RAG Chatbot with OpenRouter

A Retrieval Augmented Generation (RAG) chatbot powered by OpenRouter API for querying museum artwork data.

## Features

- 🎨 **Museum Artwork Database**: Loads and indexes artwork data from JSON files
- 🤖 **OpenRouter Integration**: Access to multiple LLM models through a single API
- 🔍 **Vector Search**: Semantic search using ChromaDB and HuggingFace embeddings
- 💬 **Conversational Chat**: Context-aware conversations with chat history
- 🐳 **Docker Support**: Easy deployment with Docker Compose
- 🌍 **Multi-language**: Support for multiple languages

## Prerequisites

- Docker and Docker Compose installed
- OpenRouter API key ([Get one here](https://openrouter.ai))

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/declared-as-ala/fizo.git
   cd fizo
   ```

2. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
   ```

3. **Build and run with Docker Compose**

   ```bash
   docker-compose up --build
   ```

4. **Access the application**

   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## Usage

1. **Enter your OpenRouter API key** in the sidebar (or use the one from `.env`)

2. **Select a model** from the dropdown (e.g., `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`)

3. **Create a vectorstore**:
   - Go to the "Create Vectorstore" tab
   - Enter a name (e.g., "museum_artworks")
   - Click "Create Vectorstore"
   - Wait for the indexing to complete (model is pre-downloaded in Docker image)

4. **Start chatting**:
   - Ask questions about the museum artworks
   - The chatbot will retrieve relevant information from the database
   - View source documents in the expandable sections

## Available Models (OpenRouter)

- `openai/gpt-4o` - OpenAI's latest model
- `openai/gpt-4o-mini` - Faster, cheaper option
- `openai/gpt-3.5-turbo` - Cost-effective option
- `anthropic/claude-3.5-sonnet` - Anthropic's latest model
- `anthropic/claude-3-haiku` - Fast Anthropic model
- `google/gemini-pro-1.5` - Google's Gemini model
- `meta-llama/llama-3.1-70b-instruct` - Open-source Llama model

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── .env                    # Create this with your API key
├── musee.oeuvres1.json     # Museum artwork data
├── RAG_chatabot_with_Langchain/
│   ├── RAG_app.py          # Main Streamlit application
│   ├── requirements.txt    # Python dependencies
│   └── data/
│       ├── tmp/            # Temporary files (excluded from git)
│       └── vector_stores/  # ChromaDB vector stores (excluded from git)
└── README.md
```

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)

### Streamlit Configuration

All configuration is done through the Streamlit sidebar:
- **Model Selection**: Choose from available OpenRouter models
- **Temperature**: Control randomness (0.0 - 2.0)
- **Top P**: Control diversity (0.0 - 1.0)
- **Language**: Select assistant language
- **Retriever Type**: Choose retrieval method

## Stopping the Application

To stop the application gracefully:

```bash
docker-compose down
```

Or press `Ctrl+C` in the terminal where `docker-compose up` is running.

## Troubleshooting

### Vectorstore not found

- Make sure you've created a vectorstore before trying to chat
- Check that the `data/vector_stores/` directory exists and is writable

### API Key issues

- Verify your OpenRouter API key is correct
- Check that the key has sufficient credits/quota

### Out of memory

- The embeddings model is pre-downloaded during Docker build (~90MB)
- Ensure Docker has at least 2GB of RAM allocated

### Model downloads on first use

If you see model downloads when creating vectorstore, rebuild the Docker image:
```bash
docker-compose build --no-cache
```

## Development

To run locally without Docker:

1. Install dependencies:
   ```bash
   pip install -r RAG_chatabot_with_Langchain/requirements.txt
   ```

2. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY=your-key-here
   ```

3. Run Streamlit:
   ```bash
   streamlit run RAG_chatabot_with_Langchain/RAG_app.py
   ```

## License

MIT License

## Support

For issues or questions, please open an issue on the repository.

