# azure_enterprise_chat_demo.py

import os
import re
import json
from datetime import datetime as pydatetime
from typing import Any, List, Dict
from dotenv import load_dotenv

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

# (Your custom functions in enterprise_functions.py)
from enterprise_functions import enterprise_fns

load_dotenv()

# 1) Create a credential/client
credential = DefaultAzureCredential()
project_client = AIProjectClient.from_connection_string(
    credential=credential,
    conn_str=os.environ["PROJECT_CONNECTION_STRING"]
)

# 2) Optionally set up Bing grounding
try:
    bing_connection = project_client.connections.get(connection_name=os.environ["BING_CONNECTION_NAME"])
    conn_id = bing_connection.id
    bing_tool = BingGroundingTool(connection_id=conn_id)
    print("bing > connected")
except Exception:
    bing_tool = None
    print("bing failed > no connection found or permission issue")

# 3) Optionally set up local doc search
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

# 4) Create a LoggingToolSet that prints each function call
class LoggingToolSet(ToolSet):
    def execute_tool_calls(self, tool_calls: List[Any]) -> List[dict]:
        """
        1) Print the name/arguments of each function call.
        2) Execute them using the parent ToolSet logic.
        3) Print the results.
        """
        for c in tool_calls:
            if hasattr(c, "function") and c.function:
                fn_name = c.function.name
                fn_args = c.function.arguments
                print(f"{fn_name} inputs > {fn_args} (id:{c.id})")

        raw_outputs = super().execute_tool_calls(tool_calls)

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

# 5) Create or reuse the agent
AGENT_NAME = "my-enterprise-agent-demo"
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
    "You have access to hr documents in file_search, the grounding engine from bing and custom python functions like "
    "fetch_weather, fetch_stock_price, send_email, etc. Provide well-structured, concise, professional answers. "
    "When you call a function like 'send_email', you must respond to the user with an appropriate success message "
    "(e.g., 'Email sent successfully!') so that they know the function was executed. "
)

if found_agent:
    agent = project_client.agents.update_agent(
        agent_id=found_agent.id,
        model=found_agent.model,
        instructions=found_agent.instructions,
        toolset=toolset
    )
    print(f"reusing agent > {agent.name} (id: {agent.id})")
else:
    agent = project_client.agents.create_agent(
        model=model_name,
        name=AGENT_NAME,
        instructions=instructions,
        toolset=toolset
    )
    print(f"created agent > {agent.name} (id: {agent.id})")

# 6) Create a new thread
thread = project_client.agents.create_thread()
print(f"thread > created (id: {thread.id})")

# 7) Event Handler for debugging
class MyEventHandler(AgentEventHandler):
    def __init__(self):
        super().__init__()
        self._current_message_id = None
        self._accumulated_text = ""

    def on_message_delta(self, delta: MessageDeltaChunk) -> None:
        if delta.id != self._current_message_id:
            if self._current_message_id is not None:
                print()  # finish the old line
            self._current_message_id = delta.id
            self._accumulated_text = ""
            print("\nassistant > ", end="")

        partial_text = ""
        if delta.delta.content:
            for chunk in delta.delta.content:
                partial_text += chunk.text.get("value", "")
        self._accumulated_text += partial_text

        print(partial_text, end="", flush=True)

    def on_thread_message(self, message: ThreadMessage) -> None:
        # If the message is completed assistant text, put a newline
        if message.status == "completed" and message.role == "assistant":
            print()
            self._current_message_id = None
            self._accumulated_text = ""
        else:
            print(f"{message.status.name.lower()} (id: {message.id})")

    def on_thread_run(self, run: ThreadRun) -> None:
        print(f"status > {run.status.name.lower()}")
        if run.status == "failed":
            print(f"error > {run.last_error}")

    def on_run_step(self, step: RunStep) -> None:
        print(f"{step.type.name.lower()} > {step.status.name.lower()}")

    def on_run_step_delta(self, delta: RunStepDeltaChunk) -> None:
        if delta.delta.step_details and delta.delta.step_details.tool_calls:
            for tcall in delta.delta.step_details.tool_calls:
                if getattr(tcall, "function", None) and tcall.function.name:
                    print(f"tool call > {tcall.function.name}")

    def on_unhandled_event(self, event_type: str, event_data):
        print(f"unhandled > {event_type} > {event_data}")

    def on_error(self, data: str) -> None:
        print(f"error > {data}")

    def on_done(self) -> None:
        print("done")

# 8) Helper functions
def extract_bing_query(request_url: str) -> str:
    match = re.search(r'q="([^"]+)"', request_url)
    if match:
        return match.group(1)
    return request_url

def convert_dict_to_chatmessage(msg: dict) -> ChatMessage:
    return ChatMessage(
        role=msg["role"],
        content=msg["content"],
        metadata=msg.get("metadata", None)
    )

# 9) The main chat function
def azure_enterprise_chat(user_message: str, history: List[dict]):
    """
    A single-run approach: all function calls happen inline (like Bing).
    We do *not* create a second run or do a 'requires_action' block.
    """

    # Convert existing messages from dict to ChatMessage
    conversation = []
    for msg_dict in history:
        conversation.append(convert_dict_to_chatmessage(msg_dict))

    # Add user's new message to conversation
    conversation.append(ChatMessage(role="user", content=user_message))
    yield conversation, ""

    # Post user message
    project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=user_message
    )

    # For partial function call data
    call_id_for_index: Dict[int, str] = {}
    partial_calls_by_index: Dict[int, dict] = {}
    partial_calls_by_id: Dict[str, dict] = {}
    in_progress_tools: Dict[str, ChatMessage] = {}

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
        if name_chunk:
            storage["name"] += name_chunk
        if arg_chunk:
            storage["args"] += arg_chunk

    def finalize_tool_call(call_id: str):
        """
        Creates or updates a ChatMessage bubble for a function call.
        (So the user sees a 'pending' bubble like "Calling fetch_weather".)
        """
        if call_id not in partial_calls_by_id:
            return
        data = partial_calls_by_id[call_id]
        fn_name = data["name"].strip()
        fn_args = data["args"].strip()
        if not fn_name:
            return

        if call_id not in in_progress_tools:
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
            msg_obj = in_progress_tools[call_id]
            msg_obj.content = fn_args or ""
            msg_obj.metadata["title"] = get_function_title(fn_name)

    def upsert_tool_call(tcall: dict):
        """
        Insert or update a pending bubble for each partial or complete function call.
        """
        t_type = tcall.get("type", "")
        call_id = tcall.get("id", "")

        # BING GROUNDING
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

        # FILE SEARCH
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

        # If not function, ignore
        elif t_type != "function":
            return

        # FUNCTION CALL PARTIALS
        index = tcall.get("index")
        new_call_id = call_id
        fn_data = tcall.get("function", {})
        name_chunk = fn_data.get("name", "")
        arg_chunk = fn_data.get("arguments", "")

        # Assign a call_id for that index
        if new_call_id:
            call_id_for_index[index] = new_call_id

        call_id = call_id_for_index.get(index)
        if not call_id:
            if index not in partial_calls_by_index:
                partial_calls_by_index[index] = {"name": "", "args": ""}
            accumulate_args(partial_calls_by_index[index], name_chunk, arg_chunk)
            return

        # Accumulate partial data
        if call_id not in partial_calls_by_id:
            partial_calls_by_id[call_id] = {"name": "", "args": ""}
        if index in partial_calls_by_index:
            old_data = partial_calls_by_index.pop(index)
            partial_calls_by_id[call_id]["name"] += old_data.get("name", "")
            partial_calls_by_id[call_id]["args"] += old_data.get("args", "")

        accumulate_args(partial_calls_by_id[call_id], name_chunk, arg_chunk)
        finalize_tool_call(call_id)

    # Use a single-run approach
    with project_client.agents.create_stream(
        thread_id=thread.id,
        agent_id=agent.id,
        event_handler=MyEventHandler()  # For console debugging
    ) as stream:
        for item in stream:
            event_type, event_data, *_ = item

            # Filter out None
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

                # In-progress tool calls
                if step_type == "tool_calls" and step_status == "in_progress":
                    for tcall in event_data["step_details"].get("tool_calls", []):
                        upsert_tool_call(tcall)
                    yield conversation, ""

                # Once all function calls are completed, we actually run them
                elif step_type == "tool_calls" and step_status == "completed":
                    # Mark the tool call bubbles as "done"
                    for cid, msg_obj in in_progress_tools.items():
                        msg_obj.metadata["status"] = "done"

                    # Actually execute each function call
                    # partial_calls_by_id = { call_id: {"name":"fetch_weather", "args":'{...}'} }
                    for cid, data in partial_calls_by_id.items():
                        fn_name = data["name"].strip()
                        fn_args_str = data["args"].strip()

                        # Attempt to parse the JSON arguments
                        try:
                            fn_args = json.loads(fn_args_str) if fn_args_str else {}
                        except:
                            fn_args = {}

                        # Do the actual function call
                        tool_result = toolset.execute_tool_calls([{
                            "id": cid,
                            "function": {
                                "name": fn_name,
                                "arguments": fn_args_str
                            }
                        }])

                        # We can just show a short bubble "I have done what you want" + the actual output
                        for r in tool_result:
                            conversation.append(ChatMessage(
                                role="assistant",
                                content=(
                                    f"I have done what you want (function: {fn_name}).\n"
                                    f"Result: {r['output']}"
                                )
                            ))

                    # Clear the partials
                    in_progress_tools.clear()
                    partial_calls_by_id.clear()
                    partial_calls_by_index.clear()
                    call_id_for_index.clear()

                    yield conversation, ""

                # If a new assistant message is created, add a bubble
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
                matching_msg = None
                for msg in reversed(conversation):
                    if (
                        msg.metadata
                        and msg.metadata.get("id") == message_id
                        and msg.role == "assistant"
                    ):
                        matching_msg = msg
                        break

                if matching_msg:
                    matching_msg.content += agent_msg
                else:
                    # if the last bubble is not an assistant or is a "tool-xxx" bubble
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
                    # Mark any in-progress tool calls as done
                    for cid, msg_obj in in_progress_tools.items():
                        msg_obj.metadata["status"] = "done"
                    in_progress_tools.clear()
                    partial_calls_by_id.clear()
                    partial_calls_by_index.clear()
                    call_id_for_index.clear()
                    yield conversation, ""

            # 5) Final done
            elif event_type == "thread.message.completed":
                # Mark all tool calls done
                for cid, msg_obj in in_progress_tools.items():
                    msg_obj.metadata["status"] = "done"
                in_progress_tools.clear()
                partial_calls_by_id.clear()
                partial_calls_by_index.clear()
                call_id_for_index.clear()
                yield conversation, ""
                break

    return conversation, ""

# 10) Build the Gradio UI
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
        return evt.value["text"]

    gr.HTML("<h1 style=\"text-align: center;\">Azure AI Agent Service</h1>")

    chatbot = gr.Chatbot(
        type="messages",
        examples=[
            {"text": "What's my company's remote work policy?"},
            {"text": "Check if it will rain tomorrow?"},
            {"text": "How is Microsoft's stock doing today?"},
            {"text": "Send my direct report an email about the new HR policy."},
        ],
        show_label=False,
        scale=1,
    )

    textbox = gr.Textbox(
        show_label=False,
        lines=1,
        submit_btn=True,
    )

    chatbot.example_select(fn=on_example_clicked, inputs=None, outputs=textbox)

    # On submit:
    (textbox
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

    chatbot.clear(fn=clear_thread, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
