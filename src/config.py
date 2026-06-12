# Define some dummy data
MODELS = ["OpenAI: gpt-3.5-turbo", "OpenAI: gpt-3.5-turbo-16k", "OpenAI: gpt-4", "OpenAI: gpt-4-1106-preview",
          "Anthropic: claude-3-opus-20240229","Google: gemini-pro"]

#the code will only use the first model listed currently
EMBEDDING_MODELS = ["all-MiniLM-L6-v2", "BM25"]

TEMPERATURE = .5

#max tokens for openai llm or other llm used in the app. you will add if condition to check if context is too long
MAX_TOKENS = 8193

APP_NAME = "I'm Lucy - helping you chat with your PDF"

# make sure to include the trailing slash
PROCESSED_DOCUMENTS_DIR = "../data/processed/"
REPORTS_DOCUMENTS_DIR = "../data/reports/"


STOP_WORD_LIST = ['what', 'are', 'is', 'the', 'of', 'in', 'to', 'and', 'a', 'for', 'on', 'with', 'how', 'that', 'by', 'as', 'from', 'this', 'at', 'an', 'be', 'it', 
                          'or', 'which', 'can', 'you', 'your', 'we', 'our', 'us', 'they', 'their', 'he', 'she', 'his', 'her', 'him', 'i', 'my', 'me', 'them', 'there', 'these', 
                          'those', 'if', 'then', 'than', 'so', 'not', 'no', 'yes', 'also', 'only', 'just', 'but', 'however', 'more', 'most', 'all', 'any', 'some', 'many', 'much', 
                          'few', 'several', 'other', 'another', 'each', 'every', 'own', 'same', 'different', 'such', 'like', 'likely', 'unlikely', 'will', 'would', 'should', 'could', 
                          'can', 'may', 'might', 'must', 'shall', 'do', 'does', 'did', 'done']