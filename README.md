# LangGraph Academy Coding

This repository contains code examples and exercises for learning LangGraph, focusing on state management, chains, and graph-based workflows.

## Project Structure

```
langgraph-academy-coding/
├── common/               # Common utilities shared across modules
│   └── image_display.py
├── module_1/            # First module - Basic Chains and Graphs
│   ├── chain/          
│   └── simple-graph/
└── module_2/            # Second module - State Management
    └── state/
```

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd langgraph-academy-coding
```

2. Install dependencies:
```bash
pip install langchain langgraph pydantic python-dotenv
```

3. Create a `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=your_project_name
```

## Usage

Each module contains different examples demonstrating various aspects of LangGraph:

- `module_1/`: Basic examples of chains and graphs
- `module_2/`: Advanced state management examples

## Environment Variables

The project uses the following environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGCHAIN_API_KEY`: Your LangChain API key
- `LANGCHAIN_PROJECT`: Your LangChain project name

Make sure to set these in your `.env` file before running the examples.