"""Holds constants."""

DEFAULT_MODEL = "gpt-4.1"

# Prices in USD as of February 2025 (https://openai.com/api/pricing/)
MODEL_TO_INPUT_PRICE_PER_TOKEN = {
    "gpt-4.1": 5.0 / 10**6,
    "gpt-4.1-mini": 0.3 / 10**6,
    "gpt-4o": 5.0 / 10**6,
    "gpt-4o-mini": 0.2 / 10**6,
    "o4": 15.0 / 10**6,
    "o4-mini": 3.0 / 10**6,
}

MODEL_TO_OUTPUT_PRICE_PER_TOKEN = {
    "gpt-4.1": 15.0 / 10**6,
    "gpt-4.1-mini": 0.6 / 10**6,
    "gpt-4o": 15.0 / 10**6,
    "gpt-4o-mini": 0.8 / 10**6,
    "o4": 60.0 / 10**6,
    "o4-mini": 12.0 / 10**6,
}

FINETUNING_MODEL_TO_INPUT_PRICE_PER_TOKEN = {
    "gpt-4.1": 4.0 / 10**6,
    "gpt-4.1-mini": 0.35 / 10**6,
}

FINETUNING_MODEL_TO_OUTPUT_PRICE_PER_TOKEN = {
    "gpt-4.1": 16.0 / 10**6,
    "gpt-4.1-mini": 0.7 / 10**6,
}

FINETUNING_MODEL_TO_TRAINING_PRICE_PER_TOKEN = {
    "gpt-4.1": 28.0 / 10**6,
    "gpt-4.1-mini": 3.5 / 10**6,
}

DEFAULT_FINETUNING_EPOCHS = 4

CONSISTENT_TEMPERATURE = 0.2
CREATIVE_TEMPERATURE = 0.8

PUBMED_TOOL_NAME = "pubmed_search"
PUBMED_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": PUBMED_TOOL_NAME,
        "description": "Get abstracts or the full text of biomedical and life sciences articles from PubMed Central.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to use to search PubMed Central for scientific articles.",
                },
                "num_articles": {
                    "type": "integer",
                    "description": "The number of articles to return from the search query.",
                },
                "abstract_only": {
                    "type": "boolean",
                    "description": "Whether to return only the abstract of the articles.",
                },
            },
            "required": ["query", "num_articles"],
        },
    },
}
