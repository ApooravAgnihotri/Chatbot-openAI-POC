import os
from dotenv import load_dotenv
import openai
import pyodbc

# Load environment variables from a .env file
load_dotenv()

# Fetch environment variables
sql_server = os.getenv('AZURE_SQL_SERVER')
sql_database = os.getenv('AZURE_SQL_DATABASE')
sql_username = os.getenv('AZURE_SQL_USERNAME')
sql_password = os.getenv('AZURE_SQL_PASSWORD')

openai_api_base = os.getenv('AZURE_OPENAI_API_BASE')
openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai_deployment_embedding = os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDING')
openai_model_embedding = os.getenv('AZURE_OPENAI_MODEL_EMBEDDING')
openai_deployment_completion = os.getenv('AZURE_OPENAI_DEPLOYMENT_COMPLETION')
openai_model_completion = os.getenv('AZURE_OPENAI_MODEL_COMPLETION')
openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION')

cogsearch_name = os.getenv('AZURE_COGSEARCH_NAME')
cogsearch_index_name = os.getenv('AZURE_COGSEARCH_INDEX_NAME')
cogsearch_api_key = os.getenv('AZURE_COGSEARCH_API_KEY')
cogsearch_endpoint = os.getenv('AZURE_COGSEARCH_ENDPOINT')

# Set OpenAI API configurations
openai.api_key = openai_api_key
openai.api_base = openai_api_base
openai.api_type = "azure"
openai.api_version = openai_api_version

# Azure SQL Database connection
def get_sql_connection():
    connection_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={sql_server};DATABASE={sql_database};UID={sql_username};PWD={sql_password}"
    conn = pyodbc.connect(connection_str)
    return conn

# Example function to use OpenAI's completion
def generate_text(prompt):
    response = openai.Completion.create(
        engine=openai_deployment_completion,
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

# Example usage
if __name__ == "__main__":
    # Test SQL connection
    try:
        conn = get_sql_connection()
        print("SQL connection successful.")
        conn.close()
    except Exception as e:
        print(f"SQL connection failed: {e}")

    # Test OpenAI completion
    prompt = "Hello, world!"
    try:
        result = generate_text(prompt)
        print("OpenAI response:", result)
    except Exception as e:
        print(f"OpenAI request failed: {e}")
