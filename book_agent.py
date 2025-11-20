import os
import json
from dotenv import load_dotenv
from typing import List, TypedDict, Optional, Annotated
from pydantic import BaseModel, Field, ValidationError

# LangGraph/LangChain Imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# Load environment variables (MUST be run first)
load_dotenv()


# ----------------------------------------------------
# A. SCHEMA & MODEL SETUP
# ----------------------------------------------------

# Define the structured output the LLM must adhere to
class BookCategory(BaseModel):
    """Represents a category of books and a mood recommendation for that category."""
    genre: str = Field(description="The broad genre of the books (e.g., Sci-Fi, Fantasy, Mystery).")
    books: List[str] = Field(description="A list of book titles belonging to this genre.")
    mood_summary: str = Field(
        description="A detailed, evocative description of the feeling or experience the user should seek to read these books (e.g., 'If you are in a mood to experience high-stakes political intrigue and immersive world-building, read these books.').")


class CategorizedBookList(BaseModel):
    """The complete list of books categorized by genre."""
    categorized_books: List[BookCategory] = Field(description="A list of all categorized groups.")


# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)


# ----------------------------------------------------
# B. AGENT STATE (MEMORY)
# ----------------------------------------------------

# Define the state/memory of the graph
class AgentState(TypedDict):
    """The state of the agent, passed between nodes."""
    book_list_raw: str
    categorized_data: Optional[CategorizedBookList]
    error_flag: bool
    chat_history: Annotated[List[BaseMessage], add_messages]


# ----------------------------------------------------
# C. NODE FUNCTIONS (4 NODES)
# ----------------------------------------------------

# NODE 1: Sets up the initial state with the collected input
def set_initial_state(state: AgentState) -> AgentState:
    """Sets the initial state based on the input collected before graph invocation."""
    return {
        "error_flag": False,
        "chat_history": [AIMessage(content=f"Received input. Starting categorization.")]
    }


# NODE 2: Performs the main LLM work
def categorizer(state: AgentState) -> AgentState:
    """Calls the LLM to categorize the books based on the raw list."""

    # 1. Initialize the Parser and Prompt
    parser = JsonOutputParser(pydantic_object=CategorizedBookList)

    system_prompt = (
        "You are an expert literary assistant. Categorize the provided list of books by genre. "
        "For each genre, write a compelling, mood-based recommendation. "
        "The mood summary MUST be written in the format: 'If you are in a mood to experience [specific experience/feeling], read these books.' "
        "Crucially, DO NOT suggest any book that is not in the provided list. "
        "Output the result ONLY as a single JSON object that strictly conforms to the following schema."
        "\n{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        HumanMessage(content=f"Here is my list of books:\n{state['book_list_raw']}")
    ])

    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser

    # 2. Invoke the Chain and Handle Parsing Errors
    try:
        raw_output = chain.invoke({"book_list": state['book_list_raw']})

        # Validation checks the structured output
        categorized_data = CategorizedBookList.parse_obj(raw_output)

        # Successful run
        return {
            "categorized_data": categorized_data,
            "error_flag": False,
        }

    except (json.JSONDecodeError, ValidationError, AttributeError) as e:
        # Failed to parse the LLM's JSON output
        return {
            "error_flag": True,
            "chat_history": [AIMessage(
                content="The AI failed to generate a clean structured output. This is often fixed by retrying or simplifying the book list.")]
        }
    except Exception as e:
        # General LLM or API failure
        return {
            "error_flag": True,
            "chat_history": [AIMessage(content=f"A general error occurred during processing: {e}")]
        }


# NODE 3: Handles processing errors
def error_handler(state: AgentState) -> AgentState:
    """Sets a final error message if categorization failed."""
    return state  # Pass the state directly to finalizer


# NODE 4: Generates the final, clear output
def finalizer(state: AgentState) -> AgentState:
    """Formats and prints the final output based on success or error."""

    if state["error_flag"] or not state.get("categorized_data"):
        # Output the error message set in the categorizer
        print("\n" + "=" * 50)
        print("    🛑 PROCESSING FAILED 🛑")
        print("=" * 50)
        print(state["chat_history"][-1].content)
        print("=" * 50)
        return state

    # Successful output formatting
    categorized_result = state["categorized_data"]

    print("\n" + "=" * 50)
    print("    ✨ YOUR ORGANIZED READING GUIDE ✨")
    print("=" * 50)

    for category in categorized_result.categorized_books:
        print(f"\n## 🌟 {category.genre.upper()} 🌟")
        print(f"> {category.mood_summary}")

        print("\n**Books in this category:**")
        for book in category.books:
            print(f"  * {book}")

    print("\n" + "=" * 50)

    return state


# ----------------------------------------------------
# D. ROUTER AND GRAPH ASSEMBLY
# ----------------------------------------------------

# Router function (Decision / Branch)
def check_for_error(state: AgentState) -> str:
    """Decides the next node based on the error flag."""
    if state["error_flag"]:
        return "ERROR"
    else:
        return "SUCCESS"


# Build the LangGraph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("set_initial_state", set_initial_state)
workflow.add_node("categorizer", categorizer)
workflow.add_node("error_handler", error_handler)
workflow.add_node("finalizer", finalizer)

# Set Entry Point
workflow.set_entry_point("set_initial_state")

# Define Edges

# 1. After input state is set, categorize
workflow.add_edge("set_initial_state", "categorizer")

# 2. After categorization, check the status (Branch)
workflow.add_conditional_edges(
    "categorizer",
    check_for_error,
    {
        "SUCCESS": "finalizer",
        "ERROR": "error_handler"
    }
)

# 3. After error handling, finalize (print the error) and END
workflow.add_edge("error_handler", "finalizer")

# 4. Finalizer marks the end of the process
workflow.add_edge("finalizer", END)

# Compile the graph
app = workflow.compile()


# ----------------------------------------------------
# E. RUN THE APPLICATION
# ----------------------------------------------------

def collect_user_books():
    """Interactively collects the book list from the user."""
    print("\n" + "=" * 50)
    print("      📖 Genre & Mood Book Organizer 📚")
    print("=" * 50)
    print("Please enter a list of books you want to read (one title per line).")
    print("Type 'done' when finished.")

    user_input_list = []

    while True:
        line = input("Book Title: ")
        if line.lower() == 'done':
            break
        # Split by comma to allow comma-separated lists on a single line
        if line.strip():
            user_input_list.extend([title.strip() for title in line.split(',') if title.strip()])

    if not user_input_list:
        print("No books entered. Exiting.")
        return None

    return "\n".join(user_input_list)


if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n--- 🛑 FATAL ERROR: GOOGLE_API_KEY not found. 🛑 ---")
        print("Please ensure your .env file is correctly configured.")
    else:
        book_list_str = collect_user_books()

        if book_list_str:
            # Initial State: all fields must be present for LangGraph
            initial_state = {
                "book_list_raw": book_list_str,
                "categorized_data": None,
                "error_flag": False,
                "chat_history": [AIMessage(content="System initialized.")]
            }

            print("\nProcessing your book list (this may take a few moments)...")

            # Running the entire graph end-to-end in a single invoke call
            try:
                # The internal node printing is suppressed, only the finalizer prints the result
                app.invoke(initial_state)
            except Exception as e:
                print(f"\n--- CRITICAL EXECUTION ERROR --- \n{e}")