import json
from typing import List, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from mistralai import UserMessage, AssistantMessage, SystemMessage as MistralSystemMessage

from config import client, MODEL_NAME

def map_langchain_to_mistral(messages: List[BaseMessage]) -> List[Any]:

    mistral_msgs = []
    
    for msg in messages:
        if isinstance(msg, SystemMessage):
            mistral_msgs.append(MistralSystemMessage(content=msg.content))
        elif isinstance(msg, HumanMessage):
            mistral_msgs.append(UserMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            mistral_msgs.append(AssistantMessage(content=msg.content))
            
    return mistral_msgs

def call_mistral(
    messages: List[BaseMessage], 
    temperature: float = 0.7,
    json_mode: bool = False
) -> str:

    mistral_messages = map_langchain_to_mistral(messages)
    
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=mistral_messages,
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else None
    )
    
    return response.choices[0].message.content

def parse_json_garbage(text: str) -> dict:

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned)