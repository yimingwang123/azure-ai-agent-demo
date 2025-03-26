# Azure AI Agent Service Enterprise Demo

# ### Import Necessary Libraries
# In this cell, we import all the libraries and modules required for the project.
# This includes Azure AI SDKs, Gradio for UI, and custom functions.

import os
import re
from datetime import datetime as pydatetime
from typing import Any, List, Dict
from dotenv import load_dotenv
import json

# (Optional) Gradio app for UI
import gradio as gr
from gradio import ChatMessage

# Azure AI Projects
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentEventHandler,
    RunStep,
    RunStepDeltaChunk,
    ThreadMessage,
    ThreadRun,
    MessageDeltaChunk,
    BingGroundingTool,
    FilePurpose,
    FileSearchTool,
    FunctionTool,
    ToolSet
)

# Your custom Python functions (for "fetch_weather","fetch_stock_price","send_email","fetch_datetime", etc.)
from enterprise_functions import enterprise_fns

load_dotenv()

# ### Create Client and Load Azure AI Foundry
# Here, we initialize the Azure AI client using DefaultAzureCredential.
# This allows us to authenticate and connect to the Azure AI service.

credential = DefaultAzureCredential()
project_client = AIProjectClient.from_connection_string(
    credential=credential,
    conn_str=os.environ["PROJECT_CONNECTION_STRING"]
)

# ### Set Up Tools (BingGroundingTool, FileSearchTool)
# In this step, we configure tools such as `BingGroundingTool` and `FileSearchTool`.
# We check for existing connections and create or reuse vector stores for document search.

try:
    bing_connection = project_client.connections.get(connection_name=os.environ["BING_CONNECTION_NAME"])
    conn_id = bing_connection.id
    bing_tool = BingGroundingTool(connection_id=conn_id)
    print("bing > connected")
except Exception:
    bing_tool = None
    print("bing failed > no connection found or permission issue")

FOLDER_NAME = "enterprise-data"
VECTOR_STORE_NAME = "hr-policy-vector-store"
all_vector_stores = project_client.agents.list_vector_stores().data
existing_vector_store = next(
    (store for store in all_vector_stores if store.name == VECTOR_STORE_NAME),
    None
)

vector_store_id = None
if existing_vector_store:
    vector_store_id = existing_vector_store.id
    print(f"reusing vector store > {existing_vector_store.name} (id: {existing_vector_store.id})")
else:
    # If you have local docs to upload
    import os
    if os.path.isdir(FOLDER_NAME):
        file_ids = []
        for file_name in os.listdir(FOLDER_NAME):
            file_path = os.path.join(FOLDER_NAME, file_name)
            if os.path.isfile(file_path):
                print(f"uploading > {file_name}")
                uploaded_file = project_client.agents.upload_file_and_poll(
                    file_path=file_path,
                    purpose=FilePurpose.AGENTS
                )
                file_ids.append(uploaded_file.id)

        if file_ids:
            print(f"creating vector store > from {len(file_ids)} files.")
            vector_store = project_client.agents.create_vector_store_and_poll(
                file_ids=file_ids,
                name=VECTOR_STORE_NAME
            )
            vector_store_id = vector_store.id
            print(f"created > {vector_store.name} (id: {vector_store_id})")

file_search_tool = None
if vector_store_id:
    file_search_tool = FileSearchTool(vector_store_ids=[vector_store_id])

# ### Combine All Tools into a ToolSet
# This step creates a custom `ToolSet` that includes all the tools configured earlier.
# It also adds a `LoggingToolSet` subclass to log the inputs and outputs of function calls.

class LoggingToolSet(ToolSet):
    def execute_tool_calls(self, tool_calls: List[Any]) -> List[dict]:
        """
        Execute the upstream calls, printing only two lines per function:
        1) The function name + its input arguments
        2) The function name + its output result
        """

        # For each function call, print the input arguments
        for c in tool_calls:
            if hasattr(c, "function") and c.function:
                fn_name = c.function.name
                fn_args = c.function.arguments
                print(f"{fn_name} inputs > {fn_args} (id:{c.id})")

        # Execute the tool calls (superclass logic)
        raw_outputs = super().execute_tool_calls(tool_calls)

        # Print the output of each function call
        for item in raw_outputs:
            print(f"output > {item['output']}")

        return raw_outputs

custom_functions = FunctionTool(enterprise_fns)

toolset = LoggingToolSet()
if bing_tool:
    toolset.add(bing_tool)
if file_search_tool:
    toolset.add(file_search_tool)
toolset.add(custom_functions)

for tool in toolset._tools:
    tool_name = tool.__class__.__name__
    print(f"tool > {tool_name}")
    for definition in tool.definitions:
        if hasattr(definition, "function"):
            fn = definition.function
            print(f"{fn.name} > {fn.description}")
        else:
            pass

# ### (Optional) Direct Azure AI Search Integration
# Skip this cell if you're using the default File Search Tool vector store approach

if any(tool.__class__.__name__ == "FileSearchTool" for tool in toolset._tools):
    print("file_search tool exists > skipping ai_search tool add")
else:
    try:
        # Get the connection ID for your Azure AI Search resource
        connections = project_client.connections.list()
        conn_id = next(
            c.id for c in connections if c.name == os.environ.get("AZURE_SEARCH_CONNECTION_NAME")
        )

        # Initialize Azure AI Search tool for direct index access
        from azure.ai.projects.models import AzureAISearchTool
        search_tool = AzureAISearchTool(
            index_connection_id=conn_id,
            index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME")
        )

        # Add the Azure AI Search tool to our toolset
        toolset.add(search_tool)
        print("azure ai search > connected directly to index")

        # Verify the tool was added by iterating through the toolset
        for tool in toolset._tools:
            tool_name = tool.__class__.__name__
            print(f"tool > {tool_name}")
            for definition in tool.definitions:
                if hasattr(definition, "function"):
                    fn = definition.function
                    print(f"{fn.name} > {fn.description}")
                else:
                    pass

    except Exception as e:
        print(f"azure ai search > skipped (no connection configured): {str(e)}")

# ### Create or Reuse the Enterprise Agent
# In this step, we create a new enterprise agent or reuse an existing one.
# The agent is configured with a model, instructions, and the toolset from the previous step.

AGENT_NAME = "ai-agent-demo"
found_agent = None
all_agents_list = project_client.agents.list_agents().data
for a in all_agents_list:
    if a.name == AGENT_NAME:
        found_agent = a
        break

model_name = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o")
instructions = (
    "You are a helpful enterprise assistant at Microsoft. "
    f"Today's date is {pydatetime.now().strftime('%A, %b %d, %Y, %I:%M %p')}. "
    "You have access to hr documents in file_search, the grounding engine from bing and custom python functions like fetch_weather, "
    "fetch_stock_price, send_email, etc. Provide well-structured, concise, and professional answers."
)

if found_agent:
    # Update the existing agent to use new tools
    agent = project_client.agents.update_agent(
        agent_id=found_agent.id,  # Changed from assistant_id to agent_id
        model=found_agent.model,
        instructions=found_agent.instructions,
        toolset=toolset,
    )
    print(f"reusing agent > {agent.name} (id: {agent.id})")
else:
    agent = project_client.agents.create_agent(
        model=model_name,
        name=AGENT_NAME,
        instructions=instructions,
        toolset=toolset
    )
    print(f"creating agent > {agent.name} (id: {agent.id})")

# ### Create a Conversation Thread
# In this step, we create a new conversation thread for the enterprise agent.
# Threads are used to manage and track conversations with the agent.

thread = project_client.agents.create_thread()
print(f"thread > created (id: {thread.id})")

# ### Define a Custom Event Handler
# Here, we define a custom event handler to manage logs and outputs for debugging.
# This handler will capture and display real-time events during the agent's operation.

class MyEventHandler(AgentEventHandler):
    def __init__(self):
        super().__init__()
        self._current_message_id = None
        self._accumulated_text = ""

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        # If a new message id, start fresh
        if delta.id != self._current_message_id:
            # First, if we had an old message that wasn't completed, finish that line
            if self._current_message_id is not None:
                print()  # move to a new line

            self._current_message_id = delta.id
            self._accumulated_text = ""
            print("\nassistant > ", end="")  # prefix for new message

        # Accumulate partial text
        partial_text = ""
        if delta.delta.content:
            for chunk in delta.delta.content:
                partial_text += chunk.text.get("value", "")
        self._accumulated_text += partial_text

        # Print partial text with no newline
        print(partial_text, end="", flush=True)

    def on_thread_message(self, message: ThreadMessage) -> None:
        # When the assistant's entire message is "completed", print a final newline
        if message.status == "completed" and message.role == "assistant":
            print()  # done with this line
            self._current_message_id = None
            self._accumulated_text = ""
        else:
            # For other roles or statuses, you can log if you like:
            print(f"{message.status.name.lower()} (id: {message.id})")

    def on_thread_run(self, run: ThreadRun) -> None:
        print(f"status > {run.status.name.lower()}")
        if run.status == "failed":
            print(f"error > {run.last_error}")

    def on_run_step(self, step: RunStep) -> None:
        print(f"{step.type.name.lower()} > {step.status.name.lower()}")

    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        # If partial tool calls come in, we log them
        if delta.delta.step_details and delta.delta.step_details.tool_calls:
            for tcall in delta.delta.step_details.tool_calls:
                if getattr(tcall, "function", None):
                    if tcall.function.name is not None:
                        print(f"tool call > {tcall.function.name}")

    def on_unhandled_event(self, event_type: str, event_data):
        print(f"unhandled > {event_type} > {event_data}")

    def on_error(self, data: str) -> None:
        print(f"error > {data}")

    def on_done(self) -> None:
        print("done")

# ### Implement the Main Chat Functions
# These functions define how user messages and tool interactions are processed.
# It uses the agent's thread to handle conversations and streams partial responses.

def extract_bing_query(request_url: str) -> str:
    """
    Extract the query string from something like:
      https://api.bing.microsoft.com/v7.0/search?q="latest news about Microsoft January 2025"
    Returns: latest news about Microsoft January 2025
    """
    match = re.search(r'q="([^"]+)"', request_url)
    if match:
        return match.group(1)
    # If no match, fall back to entire request_url
    return request_url

def convert_dict_to_chatmessage(msg: dict) -> ChatMessage:
    """
    Convert a legacy dict-based message to a gr.ChatMessage.
    Uses the 'metadata' sub-dict if present.
    """
    return ChatMessage(
        role=msg["role"],
        content=msg["content"],
        metadata=msg.get("metadata", None)
    )

def azure_enterprise_chat(user_message: str, history: List[dict]):
    """
    Accumulates partial function arguments into ChatMessage['content'], sets the
    corresponding tool bubble status from "pending" to "done" on completion,
    and also handles non-function calls like bing_grounding or file_search by appending a
    "pending" bubble. Then it moves them to "done" once tool calls complete.

    This function returns a list of ChatMessage objects directly (no dict conversion).
    Your Gradio Chatbot should be type="messages" to handle them properly.
    """

    # Convert existing history from dict to ChatMessage
    conversation = []
    for msg_dict in history:
        conversation.append(convert_dict_to_chatmessage(msg_dict))

    # Append the user's new message
    conversation.append(ChatMessage(role="user", content=user_message))

    # Immediately yield two outputs to clear the textbox
    yield conversation, ""

    # Post user message to the thread (for your back-end logic)
    project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    # Mappings for partial function calls
    call_id_for_index: Dict[int, str] = {}
    partial_calls_by_index: Dict[int, dict] = {}
    partial_calls_by_id: Dict[str, dict] = {}
    in_progress_tools: Dict[str, ChatMessage] = {}

    # Titles for tool bubbles
    function_titles = {
        "fetch_weather": "☁️ fetching weather",
        "fetch_datetime": "🕒 fetching datetime",
        "fetch_stock_price": "📈 fetching financial info",
        "send_email": "✉️ sending mail",
        "file_search": "📄 searching docs",
        "bing_grounding": "🔍 searching bing",
    }

    def get_function_title(fn_name: str) -> str:
        return function_titles.get(fn_name, f"🛠 calling {fn_name}")

    def accumulate_args(storage: dict, name_chunk: str, arg_chunk: str):
        """Accumulates partial JSON data for a function call."""
        if name_chunk:
            storage["name"] += name_chunk
        if arg_chunk:
            storage["args"] += arg_chunk

    def finalize_tool_call(call_id: str):
        """Creates or updates the ChatMessage bubble for a function call."""
        if call_id not in partial_calls_by_id:
            return
        data = partial_calls_by_id[call_id]
        fn_name = data["name"].strip()
        fn_args = data["args"].strip()
        if not fn_name:
            return

        if call_id not in in_progress_tools:
            # Create a new bubble with status="pending"
            msg_obj = ChatMessage(
                role="assistant",
                content=fn_args or "",
                metadata={
                    "title": get_function_title(fn_name),
                    "status": "pending",
                    "id": f"tool-{call_id}"
                }
            )
            conversation.append(msg_obj)
            in_progress_tools[call_id] = msg_obj
        else:
            # Update existing bubble
            msg_obj = in_progress_tools[call_id]
            msg_obj.content = fn_args or ""
            msg_obj.metadata["title"] = get_function_title(fn_name)

    def upsert_tool_call(tcall: dict):
        """
        1) Check the call type
        2) If "function", gather partial name/args
        3) If "bing_grounding" or "file_search", show a pending bubble
        """
        t_type = tcall.get("type", "")
        call_id = tcall.get("id")

        # --- BING GROUNDING ---
        if t_type == "bing_grounding":
            request_url = tcall.get("bing_grounding", {}).get("requesturl", "")
            if not request_url.strip():
                return

            query_str = extract_bing_query(request_url)
            if not query_str.strip():
                return

            msg_obj = ChatMessage(
                role="assistant",
                content=query_str,
                metadata={
                    "title": get_function_title("bing_grounding"),
                    "status": "pending",
                    "id": f"tool-{call_id}" if call_id else "tool-noid"
                }
            )
            conversation.append(msg_obj)
            if call_id:
                in_progress_tools[call_id] = msg_obj
            return

        # --- FILE SEARCH ---
        elif t_type == "file_search":
            msg_obj = ChatMessage(
                role="assistant",
                content="searching docs...",
                metadata={
                    "title": get_function_title("file_search"),
                    "status": "pending",
                    "id": f"tool-{call_id}" if call_id else "tool-noid"
                }
            )
            conversation.append(msg_obj)
            if call_id:
                in_progress_tools[call_id] = msg_obj
            return

        # --- NON-FUNCTION CALLS ---
        elif t_type != "function":
            return

        # --- FUNCTION CALL PARTIAL-ARGS ---
        index = tcall.get("index")
        new_call_id = call_id
        fn_data = tcall.get("function", {})
        name_chunk = fn_data.get("name", "")
        arg_chunk = fn_data.get("arguments", "")

        if new_call_id:
            call_id_for_index[index] = new_call_id

        call_id = call_id_for_index.get(index)
        if not call_id:
            # Accumulate partial
            if index not in partial_calls_by_index:
                partial_calls_by_index[index] = {"name": "", "args": ""}
            accumulate_args(partial_calls_by_index[index], name_chunk, arg_chunk)
            return

        if call_id not in partial_calls_by_id:
            partial_calls_by_id[call_id] = {"name": "", "args": ""}

        if index in partial_calls_by_index:
            old_data = partial_calls_by_index.pop(index)
            partial_calls_by_id[call_id]["name"] += old_data.get("name", "")
            partial_calls_by_id[call_id]["args"] += old_data.get("args", "")

        # Accumulate partial
        accumulate_args(partial_calls_by_id[call_id], name_chunk, arg_chunk)

        # Create/update the function bubble
        finalize_tool_call(call_id)

    # -- EVENT STREAMING --
    with project_client.agents.create_stream(
        thread_id=thread.id,
        agent_id=agent.id,
        event_handler=MyEventHandler()
    ) as stream:
        for item in stream:
            event_type, event_data, *_ = item

            # Remove any None items that might have been appended
            conversation = [m for m in conversation if m is not None]

            # 1) Partial tool calls
            if event_type == "thread.run.step.delta":
                step_delta = event_data.get("delta", {}).get("step_details", {})
                if step_delta.get("type") == "tool_calls":
                    for tcall in step_delta.get("tool_calls", []):
                        upsert_tool_call(tcall)
                    yield conversation, ""

            # 2) run_step
            elif event_type == "run_step":
                step_type = event_data["type"]
                step_status = event_data["status"]

                # If tool calls are in progress, new or partial
                if step_type == "tool_calls" and step_status == "in_progress":
                    for tcall in event_data["step_details"].get("tool_calls", []):
                        upsert_tool_call(tcall)
                    yield conversation, ""

                elif step_type == "tool_calls" and step_status == "completed":
                    for cid, msg_obj in in_progress_tools.items():
                        msg_obj.metadata["status"] = "done"
                    in_progress_tools.clear()
                    partial_calls_by_id.clear()
                    partial_calls_by_index.clear()
                    call_id_for_index.clear()
                    yield conversation, ""

                # elif step_type == "message_creation" and step_status == "in_progress":
                #     msg_id = event_data["step_details"]["message_creation"].get("message_id")
                #     if msg_id:
                #         conversation.append(ChatMessage(role="assistant", content=""))
                #     yield conversation, ""

                elif step_type == "message_creation" and step_status == "in_progress":
                    msg_id = event_data["step_details"]["message_creation"].get("message_id")
                    if msg_id:
                        conversation.append(ChatMessage(
                            role="assistant",
                            content="",
                            metadata={"id": msg_id}
                        ))
                    yield conversation, ""

                elif step_type == "message_creation" and step_status == "completed":
                    yield conversation, ""

            # 3) partial text from the assistant
            elif event_type == "thread.message.delta":
                agent_msg = ""
                for chunk in event_data["delta"]["content"]:
                    agent_msg += chunk["text"].get("value", "")

                message_id = event_data["id"]

                # Try to find a matching assistant bubble
                matching_msg = None
                for msg in reversed(conversation):
                    if msg.metadata and msg.metadata.get("id") == message_id and msg.role == "assistant":
                        matching_msg = msg
                        break

                if matching_msg:
                    # Append newly streamed text
                    matching_msg.content += agent_msg
                else:
                    # Append to last assistant or create new
                    if (
                        not conversation
                        or conversation[-1].role != "assistant"
                        or (
                            conversation[-1].metadata
                            and str(conversation[-1].metadata.get("id", "")).startswith("tool-")
                        )
                    ):
                        conversation.append(ChatMessage(role="assistant", content=agent_msg))
                    else:
                        conversation[-1].content += agent_msg

                yield conversation, ""

            # 4) If entire assistant message is completed
            elif event_type == "thread.message":
                if event_data["role"] == "assistant" and event_data["status"] == "completed":
                    for cid, msg_obj in in_progress_tools.items():
                        msg_obj.metadata["status"] = "done"
                    in_progress_tools.clear()
                    partial_calls_by_id.clear()
                    partial_calls_by_index.clear()
                    call_id_for_index.clear()
                    yield conversation, ""

            # 5) Final done
            elif event_type == "thread.message.completed":
                for cid, msg_obj in in_progress_tools.items():
                    msg_obj.metadata["status"] = "done"
                in_progress_tools.clear()
                partial_calls_by_id.clear()
                partial_calls_by_index.clear()
                call_id_for_index.clear()
                yield conversation, ""
                break

            # After detecting "requires_action" status
            elif event_type == "thread.run" and event_data["status"] == "requires_action":
                run_id = event_data["id"]
                # 1. Get the required action details
                tool_calls = event_data["required_action"]["submit_tool_outputs"]["tool_calls"]
                outputs = []
                
                # 2. Execute each requested tool call using your toolset directly
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                    # Execute the tool call with your existing toolset
                    tool_outputs = toolset.execute_tool_calls([{
                        "id": tool_call["id"],
                        "function": {
                            "name": function_name,
                            "arguments": tool_call["function"]["arguments"]
                        }
                    }])
                    
                    # Add the result to outputs
                    for output in tool_outputs:
                        outputs.append({
                            "tool_call_id": output["tool_call_id"],
                            "output": output["output"]
                        })
                
                # 3. Submit the results back
                project_client.agents.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run_id,
                    tool_outputs=outputs
                )
                
                # 4. Create a NEW run to continue the conversation after tool outputs
                new_run = project_client.agents.create_run(
                    thread_id=thread.id,
                    agent_id=agent.id
                )
                
                # Continue processing events
                yield conversation, ""

    return conversation, ""

# ### Build a Gradio UI
# Create a Gradio interface for interacting with the enterprise agent.
# Include a chatbot component and a text input box for user queries.

brand_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="gray",
    font=["Segoe UI", "Arial", "sans-serif"],
    font_mono=["Courier New", "monospace"],
    text_size="lg",
).set(
    button_primary_background_fill="#0f6cbd",
    button_primary_background_fill_hover="#115ea3",
    button_primary_background_fill_hover_dark="#4f52b2",
    button_primary_background_fill_dark="#5b5fc7",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#e0e0e0",
    button_secondary_background_fill_hover="#c0c0c0",
    button_secondary_background_fill_hover_dark="#a0a0a0",
    button_secondary_text_color="#000000",
    body_background_fill="#f5f5f5",
    block_background_fill="#ffffff",
    body_text_color="#242424",
    body_text_color_subdued="#616161",
    block_border_color="#d1d1d1",
    block_border_color_dark="#333333",
    input_background_fill="#ffffff",
    input_border_color="#d1d1d1",
    input_border_color_focus="#0f6cbd",
)

with gr.Blocks(theme=brand_theme, css="footer {visibility: hidden;}", fill_height=True) as demo:

    def clear_thread():
        global thread
        thread = project_client.agents.create_thread()
        return []

    def on_example_clicked(evt: gr.SelectData):
        return evt.value["text"]  # Fill the textbox with that example text

    gr.HTML("<h1 style=\"text-align: center;\">Azure AI Agent Service</h1>")

    chatbot = gr.Chatbot(
        type="messages",
        examples=[
            {"text": "What's my company's remote work policy?"},
            {"text": "Check if it will rain tomorrow?"},
            {"text": "How is Microsoft's stock doing today?"},
            {"text": "Send my direct report a summary of the HR policy."},
        ],
        show_label=False,
        scale=1,
    )

    textbox = gr.Textbox(
        show_label=False,
        lines=1,
        submit_btn=True,
    )

    # Populate textbox when an example is clicked
    chatbot.example_select(fn=on_example_clicked, inputs=None, outputs=textbox)

    # On submit: call azure_enterprise_chat, then clear the textbox
    (
        textbox
        .submit(
            fn=azure_enterprise_chat,
            inputs=[textbox, chatbot],
            outputs=[chatbot, textbox],
        )
        .then(
            fn=lambda: "",
            outputs=textbox,
        )
    )

    # A "Clear" button that resets the thread and the Chatbot
    chatbot.clear(fn=clear_thread, outputs=chatbot)

# Launch your Gradio app
if __name__ == "__main__":
    demo.launch()

# ### (Optional) delete agent, thread, and vector store resources
# Uncomment out the next block to delete the resources created in this notebook.

# from azure.identity import DefaultAzureCredential
# from azure.ai.projects import AIProjectClient
# import os
#
# credential = DefaultAzureCredential()
# project_client_delete = AIProjectClient.from_connection_string(
#     credential=credential,
#     conn_str=os.environ.get("PROJECT_CONNECTION_STRING")
# )
#
# try:
#     project_client_delete.agents.delete_agent(agent.id)
#     print("Agent deletion successful.")
#     project_client_delete.agents.delete_thread(thread.id)
#     print("Thread deletion successful.")
#     project_client_delete.agents.delete_vector_store(vector_store_id)
#     print("Vector store deletion successful.")
#     print("All deletions succeeded.")
# except Exception as e:
#     print(f"Error during deletion: {e}")
