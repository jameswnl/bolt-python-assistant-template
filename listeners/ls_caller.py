import os
import re
from typing import List, Dict

import openai
import json

from llama_stack_client.types.agents import AgentTurnResponseStreamChunk
from llama_stack_client import AsyncLlamaStackClient, LlamaStackClient
from llama_stack_client.lib.agents.agent import AsyncAgent, Agent
from llama_stack_client.types.agent_create_params import AgentConfig


DEFAULT_SYSTEM_CONTENT = """
You're an assistant in a Slack workspace.
Users in the workspace will ask you to help them write something or to think better about a specific topic.
You'll respond to those questions in a professional way.
When you include markdown text, convert them to Slack compatible ones.
When a prompt has Slack's special syntax like <@USER_ID> or <#CHANNEL_ID>, you must keep them as-is in your response.
"""

client = LlamaStackClient(
    base_url="http://localhost:8321",
)
agent_config = AgentConfig(
    model="anthropic/claude-3-5-haiku-latest",
    # model="llama3.2:3b-instruct-fp16",
    instructions="You are a helpful Ansible Automation Platform assistant.",
    sampling_params={
        "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
    },
    toolgroups=(
        [
            # "mcp::weather",
            # "mcp::github",
            # "mcp::fs",
            "mcp::aap_api",
            # "mcp::controller_api",
            # "mcp::gateway_api",
            "builtin::websearch",
        ]
    ),
    tool_choice="auto",
    input_shields=[],  # available_shields if available_shields else [],
    output_shields=[],  # available_shields if available_shields else [],
    enable_session_persistence=False,
)
agent = Agent(client, agent_config)
session_id = agent.create_session("lightspeed-session")

def call_ls(
    messages: str,
    system_content: str = DEFAULT_SYSTEM_CONTENT,
) -> str:
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": messages,
            }
        ],
        stream=True,
        session_id=session_id,
    )
    reply_message = ""
    for chunk in response:
        j = chunk.model_dump_json()
        o = json.loads(j)  # dump to python dict # TODO
        event = o.get("event")
        if event:
            payload = event.get("payload")
            if payload:
                event_type = payload.get("event_type")
                if event_type == "step_start":
                    d = {"event": "start", "data": {"conversation_id": payload.get("step_id")}}
                    # yield self.format_record(d)
                elif event_type == "step_progress":
                    delta = payload.get("delta", [])
                    if delta:
                        delta_type = delta.get("type", "")
                        if delta_type == "text":
                            # yield self.format_token(delta.get("text", ""), id)
                            reply_message = reply_message + delta.get("text", "")
                            
                        elif delta_type == "tool_call":
                            # yield self.format_token("\n```\n", id)
                            
                            tool_call = delta.get("tool_call", "")
                            if not isinstance(tool_call, str):
                                tool_call = json.dumps(tool_call, indent=2)
                            # yield self.format_token(tool_call, id)
                            
                elif event_type == "step_complete":
                    # if not is_monospace:
                    #     is_monospace = True
                    #     yield self.format_token("\n```\n", id)
                    
                    pass
                    # step_details = payload.get("step_details")

                elif event_type == "turn_complete":
                    return markdown_to_slack(reply_message)
        
    return ""
    # return markdown_to_slack(response.choices[0].message.content)


# Conversion from OpenAI markdown to Slack mrkdwn
# See also: https://api.slack.com/reference/surfaces/formatting#basics
def markdown_to_slack(content: str) -> str:
    # Split the input string into parts based on code blocks and inline code
    parts = re.split(r"(?s)(```.+?```|`[^`\n]+?`)", content)

    # Apply the bold, italic, and strikethrough formatting to text not within code
    result = ""
    for part in parts:
        if part.startswith("```") or part.startswith("`"):
            result += part
        else:
            for o, n in [
                (
                    r"\*\*\*(?!\s)([^\*\n]+?)(?<!\s)\*\*\*",
                    r"_*\1*_",
                ),  # ***bold italic*** to *_bold italic_*
                (
                    r"(?<![\*_])\*(?!\s)([^\*\n]+?)(?<!\s)\*(?![\*_])",
                    r"_\1_",
                ),  # *italic* to _italic_
                (r"\*\*(?!\s)([^\*\n]+?)(?<!\s)\*\*", r"*\1*"),  # **bold** to *bold*
                (r"__(?!\s)([^_\n]+?)(?<!\s)__", r"*\1*"),  # __bold__ to *bold*
                (r"~~(?!\s)([^~\n]+?)(?<!\s)~~", r"~\1~"),  # ~~strike~~ to ~strike~
            ]:
                part = re.sub(o, n, part)
            result += part
    return result
