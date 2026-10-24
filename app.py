
import streamlit as st

from main import agent


# =========================================================
# 1. Page Configuration
# =========================================================

st.set_page_config(
    page_title="LangChain AI Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 LangChain AI Agent")
st.caption("Powered by LangChain, OpenAI, Tavily & WeatherStack")


# =========================================================
# 2. Session State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 3. Sidebar
# =========================================================

with st.sidebar:
    st.header("Settings")

    st.write("**Model:** GPT-4o")
    st.write("**Tools:**")
    st.write("- Tavily Web Search")
    st.write("- WeatherStack")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# 4. Helper: Extract Text from Model Output
# =========================================================

def extract_text(content) -> str:
    """Extract readable text from LangChain message content."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))

            elif isinstance(block, str):
                parts.append(block)

        return "".join(parts)

    return ""


# =========================================================
# 5. Display Chat History
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# 6. Chat Input
# =========================================================

if user_query := st.chat_input(
    "Ask about latest news, weather, or anything else..."
):

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        trace_lines = []

        with st.expander(
            "Agent Actions & Observations",
            expanded=True,
        ):
            trace_placeholder = st.empty()

        response_text = ""

        def update_trace(line: str):
            trace_lines.append(line)
            trace_placeholder.code(
                "\n".join(trace_lines),
                language="text",
            )

        def update_response():
            response_placeholder.markdown(
                response_text + "▌"
            )

        try:
            inputs = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_query,
                    }
                ]
            }

            # Stream model tokens and agent execution updates
            for mode, chunk in agent.stream(
                inputs,
                stream_mode=["messages", "updates"],
            ):

                # -----------------------------------------
                # A. Token-by-token model streaming
                # -----------------------------------------

                if mode == "messages":
                    token, metadata = chunk

                    if metadata.get("langgraph_node") != "model":
                        continue

                    content = extract_text(
                        getattr(token, "content", "")
                    )

                    if content:
                        response_text += content
                        update_response()

                # -----------------------------------------
                # B. Agent actions and tool observations
                # -----------------------------------------

                elif mode == "updates":
                    for node, update in chunk.items():

                        for message in update.get("messages", []):

                            # Tool calls
                            tool_calls = getattr(
                                message,
                                "tool_calls",
                                [],
                            )

                            if tool_calls:
                                for action in tool_calls:
                                    update_trace(
                                        f"ACTION\n"
                                        f"Tool: {action['name']}\n"
                                        f"Action Input: {action['args']}\n"
                                        f"{'-' * 40}"
                                    )

                            # Tool execution results
                            elif getattr(
                                message, "type", ""
                            ) == "tool":
                                update_trace(
                                    f"OBSERVATION\n"
                                    f"Tool: {message.name}\n"
                                    f"Result: {message.content}\n"
                                    f"{'-' * 40}"
                                )

            # Final response
            if not response_text:
                response_text = (
                    "The agent finished without returning "
                    "a text response."
                )

            response_placeholder.markdown(response_text)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                }
            )

        except Exception as exc:
            error_message = f"Agent error: {exc}"
            response_placeholder.error(error_message)
