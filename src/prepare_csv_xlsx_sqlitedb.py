from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import AgentType
from utils.prepare_sqlitedb_from_csv_xlsx import PrepareSQLFromTabularData
from utils.load_config import LoadConfig

APPCFG = LoadConfig()

if __name__ == "__main__":
    prep_sql_instance = PrepareSQLFromTabularData(APPCFG.stored_csv_xlsx_directory)
    prep_sql_instance.run_pipeline()
