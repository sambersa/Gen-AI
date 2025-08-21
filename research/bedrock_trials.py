from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage
import boto3
import streamlit as st

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-north-1"
)

# Your Claude Sonnet 4 inference profile ARN
inference_profile_arn = "arn:aws:bedrock:eu-north-1:486408064530:inference-profile/eu.anthropic.claude-3-7-sonnet-20250219-v1:0"

# ChatBedrock LLM with inference profile ARN
llm = ChatBedrock(
    model_id=inference_profile_arn,
    provider="anthropic",  # Required when using ARN
    client=bedrock_client,
    model_kwargs={"temperature": 0.7}
)

st.title("Claude Sonnet 4 Chat Test")

language = st.sidebar.selectbox("Language", ["english", "norwegian", "russian", "bengali"])
user_text = st.sidebar.text_area("What would you like to ask?", max_chars=500)

if user_text:
    messages = [[HumanMessage(content=f"Language: {language}\n{user_text}")]]  # list of lists
    response = llm.generate(messages)
    output_text = response.generations[0][0].text
    st.write(output_text)