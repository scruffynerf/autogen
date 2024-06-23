import os
import json
from typing import Dict, Optional, Union
from datetime import datetime, date

from autogen.agentchat.assistant_agent import ConversableAgent
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability
from autogen.oai.client_utils import should_hide_tools, validate_parameter

import fix_busted_json as jsonparse

class ToolsforToolless(AgentCapability):
    """
    ToolsforToolless adds tool calling to LLM services which do not otherwise support tool calling.
    This may include partial supporting OpenAI apis, but not implmenting the tool calls functionality,
    or it might be the model isn't listed as supporting tools, and so the API doesn't return these.
    It's meant to add on 'similar' functionality, in a pre/post processing method.
    When adding ToolsforToolless to an agent, the following are modified:
    - The agent's system message is appended with a note about the agent's new ability in general.
    - 2 hooks is added to the agent:
      - the `process_all_messages_before_reply` hookable method, to adjust for model templates which fail 'tool' as a role, and/or
        require a strict user/assistant/user/assistant message order. The hook potentially turns 'tool' to 'user' for messages,
        and condenses any multiple messages from the user to only one message with all contents.  This is only for this agent,
        and the new adjusted set of messages is passed onward.  Other agents which cannot handle this might have an issue.
      - the 'process_message_before_send' hookable method, processes the raw text message generated, parses for matching json 
        based calls of only the tools available for the agent to use, and adds tool call(s) to the message,
        as if the original client/API supported it.
    - A 'register_reply' function is added to the agent, to add specific tool calling info to the agent's system prompt, 
        allowing for adjusting the allowed tool list each time, this happens BEFORE the LLM call.
      This uses the existing tool registration info in the agent's config, so any adding or removing should be done there.
    Added tool calls do propagate into the stored message history, it's as if they were provided by default.
    """

    def __init__(
        self,
        verbosity: Optional[int] = 0,
        max_num_retrievals: Optional[int] = 10,
        llm_config: Optional[Union[Dict, bool]] = None,
    ):
        """
        Args:
            verbosity (Optional, int): # 0 (default) for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
            reset_db (Optional, bool): True to clear the DB before starting. Default False.
            path_to_db_dir (Optional, str): path to the directory where this particular agent's DB is stored. Default "./tmp/teachable_agent_db"
            recall_threshold (Optional, float): The maximum distance for retrieved memos, where 0.0 is exact match. Default 1.5. Larger values allow more (but less relevant) memos to be recalled.
            max_num_retrievals (Optional, int): The maximum number of memos to retrieve from the DB. Default 10.
            llm_config (dict or False): llm inference configuration passed to TextAnalyzerAgent.
                If None, TextAnalyzerAgent uses llm_config from the teachable agent.
        """
        self.verbosity = verbosity
        self.max_num_retrievals = max_num_retrievals
        self.llm_config = llm_config

        self.toolsfortoolless_agent = None
        self.tool_use_system_msg = "\nYou can use these tools using json:\n{\"name\": \"toolname\", \"arguments\": {\"arg1\": arg1, \"arg2\": arg2}}'\n"
        self.tool_list_start_msg = "Available Tools: "
        self.tool_list_end_msg = ""
        self.no_tools_available = "There are NO tools available for you at this time."
        self.tool_call_template = f"Tool name: {item['function']['name']}\n"

        self.tool_self_executing = False 

    def add_to_agent(self, agent: ConversableAgent):
        """Adds ToolsforToolless to the given agent."""
        self.toolsfortoolless_agent = agent

        # Register a hook for processing the last message.
        agent.register_hook(hookable_method="process_last_received_message", hook=self.process_last_received_message)

        # Append extra info to the system message.
        agent.update_system_message(agent.system_message + self.tool_use_system_msg)
    
    def add_tools(self, recipient, messages, sender, config):
        """
        adds a list of tools to the system prompt
        """
        # hook to reduce tools?
        # should_hide_tools(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], hide_tools_param: str) -> bool:
        #Determines if tools should be hidden. This function is used to hide tools when they have been run, minimising the chance of the LLM choosing them when they shouldn't.
        #Parameters:
        #messages (List[Dict[str, Any]]): List of messages
        #tools (List[Dict[str, Any]]): List of tools
        #hide_tools_param (str): "hide_tools" parameter value. Can be "if_all_run" (hide tools if all tools have been run), "if_any_run" (hide tools if any of the tools have been run), "never" (never hide tools). Default is "never".

        # tool choice in config?

        current_system_message = recipient.system_message
        if tool_schema := recipient.llm_config["tools"]:
            # we need to update the list to current tools
            if self.tool_list_start_msg in current_system_message:
                # we need to remove the existing list:
                current_system_message = clean_old_tools(self,current_system_message)
            active_tool_list = []
            for item in tool_schema:
                        tool_list.append(item['function']['name'])
                        result += f"Tool name: {item['function']['name']}\n"
                        result += f"Tool description: {item['function']['description']}\n"
                        result += f"     arguments: {item['function']['parameters']['properties']}\n"
            recipient.update_system_message(f"{recipient.system_message}\n\nAvailable Tools: {tool_list}\n{result}")
        else:
            # there are no tools
            current_system_message = clean_old_tools(self,current_system_message)
            recipient.update_system_message(current_system_message + self.no_tools_available)
        return False, None  # required to ensure the agent communication flow continues

    def clean_old_tools(self,system_message):
        #remove the old section TBD
        return

    def parse_tool_calls(self, response_text, tool_list):
        """
        parses thru the LLM response before it is sent onward and adds tool_calls to the message
        The call must be one of the tools in tool_list (list of functions valid to use)
        jsonparse is currently using 'fix_broken_json' python library, but could be replaced.
        `pip install fix_broken_json` if needed
        """
        tool_calls = []
        # make sure we have both text, and there is at least one bracket
        # you could also tighten this to require more delineation like [TOOLCALL] {...}
        if response_text and "{" in response_text:
            content_parsed = jsonparse.to_array_of_plain_strings_or_json(response_text)
            for section in content_parsed:
                # is it json...
                if jsonparse.can_parse_json(section):
                    try:
                        tool_call_data = json.loads(section)
                        #print(f"parsed: {tool_call_data}")
                        if tool_call_data.get('name') in tool_list and tool_call_data.get('arguments'):
                            tool_call_id = f"toolcall-{int(datetime.now().timestamp())}"
                            tool_call = {'id': tool_call_id, 'type': 'function'}
                            function_dict = {
                                "name": tool_call_data['name'],
                                "arguments": json.dumps(tool_call_data['arguments'])
                            }
                            tool_call['function'] = function_dict
                            tool_calls.append(tool_call)
                    except json.JSONDecodeError:
                        #print(f"error parsing: {tool_call_section}")
                        pass
        return tool_calls, response_text

    def recognize_tool_usage(self, sender, message, recipient, silent):
        """
        function to be hooked into register_reply, which calls the parser which adds tool_calls
        to the message if any
        """
        # be sure we get a bare message... string only
        if isinstance(message, str):
            if tool_schema := sender.llm_config["tools"]:
                tool_list = [item['function']['name'] for item in tool_schema]
                tool_calls, fullmessage = parse_tool_calls(message, tool_list)
                if tool_calls:
                    return {
                        "role": 'assistant',
                        "function_call": None,
                        "content": message,
                        "tool_calls": tool_calls,
                }
        return message

    def fix_message_ownership(self, messages):
        """ 
        Some model templates hate any roles other than user or assistant or system
        and won't work (templatewise) if it's not a ping-pong conversation 
        (so two user messages in a row is still a fail, so just converting 'tool' to user is not enough)
        """
        cleanmessages = []
        previous_was_user = False
        for message in messages:
            #print(f"{message}")
            if message['role'] in ['tool','user']:
                if previous_was_user:
                    # we must merge... avoid two user messages in row
                    cleanmessages[-1]['content'] += message['content']
                    if message.get('tool_responses'):
                        for response in message['tool_responses']:
                            response['role'] = 'user'
                        if cleanmessages[-1].get('tool_responses'):
                            cleanmessages[-1]['tool_responses'].append(message.get('tool_responses'))
                        else:
                            cleanmessages[-1]['tool_responses'] = message.get('tool_responses')
                else:
                    # previous wasn't user so we are now
                    previous_was_user = True
                    message['role'] = 'user'
                    if message.get('tool_responses'):
                        for response in message['tool_responses']:
                            response['role'] = 'user'
                    cleanmessages.append(message)
            else:
                # this isn't user, so clear the flag
                previous_was_user = False
                cleanmessages.append(message)
        return cleanmessages
