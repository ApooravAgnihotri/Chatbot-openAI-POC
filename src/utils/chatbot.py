import os
import sys
from typing import List, Tuple
from load_config import LoadConfig
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentType
import langchain

# Enable debugging
langchain.debug = True

# Adding the parent directory to the system path to locate the utils package
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from load_config import LoadConfig

# Load application configuration
APPCFG = LoadConfig()

class ChatBot:
    """
    A ChatBot class capable of responding to messages using different modes of operation.
    It can interact with SQL databases, leverage language chain agents for Q&A,
    and use embeddings for Retrieval-Augmented Generation (RAG) with ChromaDB.
    """

    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        """
        Respond to a message based on the given chat and application functionality types.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Describes the type of the chat (interaction with SQL DB or RAG).
            app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

        Returns:
            Tuple[str, List, Optional[Any]]: A tuple containing an empty string, the updated chatbot conversation list,
                                             and an optional 'None' value.
        """
        if app_functionality == "Chat":
            if chat_type == "Q&A with stored SQL-DB":
                if os.path.exists(APPCFG.sqldb_directory):
                    db = SQLDatabase.from_uri(f"sqlite:///{APPCFG.sqldb_directory}")
                    execute_query = QuerySQLDataBaseTool(db=db)
                    write_query = create_sql_query_chain(APPCFG.langchain_llm, db)
                    answer_prompt = PromptTemplate.from_template(APPCFG.agent_llm_system_role)
                    answer = answer_prompt | APPCFG.langchain_llm | StrOutputParser()
                    chain = (
                        RunnablePassthrough.assign(query=write_query).assign(
                            result=itemgetter("query") | execute_query
                        )
                        | answer
                    )
                    response = chain.invoke({"question": message})
                else:
                    chatbot.append((message, "SQL DB does not exist. Please first create the 'sqldb.db'."))
                    return "", chatbot, None

            elif chat_type in ["Q&A with Uploaded CSV/XLSX SQL-DB", "Q&A with stored CSV/XLSX SQL-DB"]:
                if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                    else:
                        chatbot.append((message, "SQL DB from the uploaded csv/xlsx files does not exist. Please first upload the csv files from the chatbot."))
                        return "", chatbot, None

                elif chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.stored_csv_xlsx_sqldb_directory):
                        engine = create_engine(f"sqlite:///{APPCFG.stored_csv_xlsx_sqldb_directory}")
                    else:
                        chatbot.append((message, "SQL DB from the stored csv/xlsx files does not exist. Please first execute `src/prepare_csv_xlsx_sqlitedb.py` module."))
                        return "", chatbot, None

                db = SQLDatabase(engine=engine)
                agent_executor = create_sql_agent(
                    llm=APPCFG.langchain_llm,
                    toolkit=SQLDatabaseToolkit(db=db, llm=APPCFG.langchain_llm),
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
                )
                response = agent_executor.invoke({"input": message})["output"]

            elif chat_type == "RAG with stored CSV/XLSX ChromaDB":
                response = APPCFG.azure_openai_client.embeddings.create(
                    input=message,
                    model=APPCFG.embedding_model_name
                )
                query_embeddings = response.data[0].embedding
                vectordb = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
                results = vectordb.query(query_embeddings=query_embeddings, n_results=APPCFG.top_k)
                prompt = f"User's question: {message} \n\n Search results:\n {results}"

                messages = [
                    {"role": "system", "content": str(APPCFG.rag_llm_system_role)},
                    {"role": "user", "content": prompt}
                ]
                llm_response = APPCFG.azure_openai_client.chat.completions.create(
                    model=APPCFG.model_name,
                    messages=messages
                )
                response = llm_response.choices[0].message.content

            chatbot.append((message, response))
            return "", chatbot
        else:
            pass
