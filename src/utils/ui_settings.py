import gradio as gr
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import AgentType

class UISettings:
    """
    Utility class for managing UI settings.

    This class provides static methods for toggling UI components, such as a sidebar.
    """
    @staticmethod
    def toggle_sidebar(state):
        """
        Toggle the visibility state of a UI component.

        Parameters:
            state: The current state of the UI component.

        Returns:
            Tuple: A tuple containing the updated UI component state and the new state.
        """
        state = not state
        return gr.update(visible=state), state

    @staticmethod
    def feedback(data: gr.LikeData):
        """
        Process user feedback on the generated response.

        Parameters:
            data (gr.LikeData): Gradio LikeData object containing user feedback.
        """
        if data.liked:
            print("You upvoted this response: " + data.value)
        else:
            print("You downvoted this response: " + data.value)
