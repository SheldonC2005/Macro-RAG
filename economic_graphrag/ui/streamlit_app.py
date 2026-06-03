# economic_graphrag/ui/streamlit_app.py
import streamlit as st
import json
from pathlib import Path

# Add the package root (economic_graphrag) to Python path so absolute imports work
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the workflow using the absolute package name
from economic_graphrag.main_workflow import MainWorkflow

st.set_page_config(layout="wide", page_title="Economic GraphRAG")

st.title("📈 Agentic Hybrid GraphRAG for Macroeconomic Intelligence")
st.caption("A local-first RAG system using LangGraph, Ollama, Neo4j, and ChromaDB.")

# Initialize the workflow in session state
if 'workflow' not in st.session_state:
    with st.spinner("Initializing RAG workflow..."):
        st.session_state.workflow = MainWorkflow()
        st.success("Workflow initialized!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("View Details"):
                st.json(message["details"])

# Main chat input
if prompt := st.chat_input("Ask a macroeconomic question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the query with the RAG workflow
    with st.chat_message("assistant"):
        st.markdown("🧠 **Thinking...**")
        
        # Create columns for streaming logs and results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Execution Trace")
            trace_container = st.empty()
            trace_container.info("Planner > Retriever > Answer Generator")

        with col2:
            st.subheader("Live Context")
            context_container = st.empty()
            context_container.text("Waiting for retrieval...")

        try:
            workflow = st.session_state.workflow
            
            # We can't easily stream from LangGraph invoke, so we'll simulate the flow
            # and update the UI at each major step.
            
            # 1. Planning
            trace_container.info("Executing Planner Agent...")
            plan = workflow.planner.plan(prompt)
            trace_container.success("Planner executed.")
            context_container.json({"Plan": plan})

            # 2. Retrieval
            trace_container.info("Executing Retriever Agents...")
            vector_results = workflow.vector_agent.retrieve({"query": prompt}).get("vector_results", [])
            graph_results = workflow.graph_agent.retrieve({"query": prompt, "plan": plan}).get("graph_results", [])
            
            retrieved_context = {
                "vector_results": vector_results,
                "graph_results": graph_results
            }
            context_container.json(retrieved_context)
            trace_container.success("Retrievers executed.")

            # 3. Answer Generation
            trace_container.info("Executing Answer Agent...")
            final_answer_data = workflow.answer_agent.generate_answer(prompt, retrieved_context)
            trace_container.success("Answer generated!")

            # Display the final answer
            answer = final_answer_data.get("answer", "No answer could be generated.")
            sources = final_answer_data.get("sources", [])
            explainability = final_answer_data.get("explainability", "")

            st.markdown(answer)
            
            if sources:
                st.markdown("**Sources:**")
                for source in sources:
                    st.markdown(f"- {source}")
            
            with st.expander("Show Explainability & Full Context"):
                st.write("**Explainability:**", explainability)
                st.write("**Retrieved Vector Chunks:**")
                st.json(vector_results)
                st.write("**Retrieved Graph Paths:**")
                st.json(graph_results)

            # Add full response to message history for display
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "details": {
                    "sources": sources,
                    "explainability": explainability,
                    "retrieved_context": retrieved_context,
                    "plan": plan
                }
            }
            st.session_state.messages.append(assistant_message)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})

# To run this app:
# 1. Make sure all dependencies are installed: pip install -r requirements.txt
# 2. Make sure Ollama and Neo4j are running.
# 3. Run the data ingestion script if you haven't already.
# 4. In your terminal, run: streamlit run economic_graphrag/ui/streamlit_app.py
