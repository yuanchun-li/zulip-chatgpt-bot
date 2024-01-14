import os
import sys
import logging
import re
from openai import OpenAI
import zulip
from dotenv import load_dotenv
import tiktoken
import sqlite3
import datetime
import json
import requests

# Load the .env file
load_dotenv()

# Set up logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

if not os.path.exists('data'):
    os.makedirs('data')
if os.path.isfile('data.db'):
    os.rename('data.db', 'data/data.db')

db_file = 'data.db' if os.path.isfile('data.db') else 'data/data.db'
conn = sqlite3.connect(db_file)
cur = conn.cursor()

# Set up GPT API key
# openai.api_key = os.environ['OPENAI_API_KEY']
openai_api_url = os.environ['OPENAI_API_URL']
openai_api_key = os.environ['OPENAI_API_KEY']

# Set up Zulip client
client = zulip.Client(config_file=".zuliprc")

PERMISSIONS_SET_CONTEXT = os.environ['PERMISSIONS_SET_CONTEXT']
DEFAULT_MODEL_NAME = os.environ['DEFAULT_MODEL_NAME']
BOT_NAME = os.environ['BOT_NAME']
VERSION = "1.2.0"
CONTEXT_CLEARED = "context cleared!"

contexts = {}


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    # note: future models may deviate from this
    if model.startswith("gpt-3") or model.startswith("gpt-4"):
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def send_reply(reply, message):
    if message['type'] == 'private':
        response = {
            'type': 'private',
            'to': message['sender_email'],
            'content': reply,
        }
    else:
        response = {
            'type': 'stream',
            'to': message['display_recipient'],
            'subject': message['subject'],
            'content': reply,
        }
    client.send_message(response)

openai_client = OpenAI(
    base_url=openai_api_url,
    # This is the default and can be omitted
    api_key=openai_api_key
)

def get_gpt_response(messages, model=DEFAULT_MODEL_NAME):
    try:
        completion = openai_client.chat.completions.create(
            messages=messages,
            model=model,
            timeout=15
        )
        response = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        # return response, prompt_tokens, completion_tokens
        reply = f'{response}\n(prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens})'
        return reply
    except Exception as e:
        logging.error(e)
        return "GPT API error. Please try again later."

def print_help(msg):
    # return multiline string with help message
    help_message = """# GPT Bot of MobileLLM Team @ THU-AIR
This is a chatbot assistant that uses OpenAI's ChatGPT API to generate responses to your messages.

## How to use

To use the bot, simply mention it in a message, e.g. @**{bot}** hello!. The bot will then generate a response and send it back to you.
You can also write a private message to the bot without mentioning it.

## Subcommands

Subcommands are words starting with an exclamation mark, e.g. `!new`.
You can use the following subcommands to control the bot:

### General:
- `!help` - show this help message

### Context:
- `!continue` - continue the previous conversation (the bot will treat each request as a new conversation by default)
- `!clear` - clear the context (the messages prior to this will not be included in '!continue' mode)
- `!contexts` - list all available contexts (e.g. `!cicada`, `!frankie`) and their values

Example custom defined context: `!cicada` - add system context for Cicada; this may provide more accurate responses

### Model (default depends on server settings):
- `!gpt3` - use GPT-3.5 Turbo (4K tokens limit)
- `!gpt4` - use GPT-4 (128K tokens limit, please be careful due to the high price)

### Global settings:
- `!set` - (not implemented yet) show current settings
- `!set context <name> <value> - upsert a context like !cicada. Example: `!set context cicada Cicada is a business wallet`
- `!unset context <name>` - delete a context

## Example usage
- `@{bot} !gpt4 !continue Can you summarise previous messages?` - use GPT-4 and context from the current conversation
- `@{bot} !new I have a question...` - start a new conversation using GPT-3.5 and no context (previous messages will be ignored)

Bot version: {version}
""".format(bot=BOT_NAME, version=VERSION)
    send_reply(help_message, msg)


def get_subcommands(content):
    content_chunks = content.strip().split()
    subcommands = [word.lower().replace("!", "")
                   for word in content_chunks if word.startswith("!")]
    return subcommands


def remove_subcommands(content, subcommands):
    for subcommand in subcommands:
        content = re.sub(f"!{subcommand} ", "", content,
                         flags=re.IGNORECASE).strip()
        content = re.sub(f"!{subcommand}", "", content,
                         flags=re.IGNORECASE).strip()
    return content


def with_previous_messages(client, msg, messages, subcommands, token_limit, append_after_index):
    if msg['type'] == 'private':
        query = {
            'anchor': msg['id'],
            'num_before': 100,  # adjust this value as needed
            'num_after': 0,
            'apply_markdown': False,
            'include_anchor': False,
            'narrow': [{'operand': msg['sender_email'], 'operator': 'pm-with'}],
        }
    else:
        narrow = [
            {'operand': msg['display_recipient'], 'operator': 'stream'},
        ]

        # filter to topic by default
        if ("stream" not in subcommands):
            narrow.append({'operand': msg['subject'], 'operator': 'topic'})

        query = {
            'anchor': msg['id'],
            'num_before': 100,  # adjust this value as needed
            'num_after': 0,
            'apply_markdown': False,
            'include_anchor': False,
            'narrow': narrow,
        }

    previous_messages = client.get_messages(query)['messages']
    previous_messages.reverse()
    print(previous_messages)

    new_messages = messages.copy()

    for msg in previous_messages:
        content = msg['content'].strip()

        # remove mentions of the bot
        content = re.sub("@\*\*{bot}\*\*".format(bot=BOT_NAME), "", content)
        # remove token statistics
        content = re.sub("\n\(tokens: prompt=.+\)$", "", content)
        content = content.strip()

        # get subcommands (words starting with exclamation mark)
        subcommands = get_subcommands(content)

        if 'clear' in subcommands:
            break

        # don't remove in previous messages for now, as it breaks with some code blocks
        # content = remove_subcommands(content, subcommands)

        if client.email == msg['sender_email']:
            role = "assistant"
            if content == CONTEXT_CLEARED:
                break
        else:
            role = "user"

        new_messages.insert(append_after_index, {"role": role, "content": content.strip()})
        tokens = num_tokens_from_messages(messages=new_messages)

        if tokens > token_limit:
            # remove message from index 1
            new_messages = new_messages[:append_after_index] + \
                new_messages[append_after_index+1:]
            break

    return new_messages


def is_admin(client, msg):
    member = client.get_user_by_id(msg['sender_id'])
    return member.get("user", {}).get("is_admin")


def upsert_context(context_name, context_value):
    context_exists = cur.execute(
        "SELECT * FROM contexts WHERE name = ?", (context_name,)).fetchone()
    if context_exists:
        cur.execute("UPDATE contexts SET value = ? WHERE name = ?",
                    (context_value, context_name))
    else:
        cur.execute("INSERT INTO contexts (name, value) VALUES (?, ?)",
                    (context_name, context_value))
    conn.commit()
    refetch_contexts()


def delete_context(context_name):
    cur.execute("DELETE FROM contexts WHERE name = ?", (context_name,))
    conn.commit()
    refetch_contexts()


def refetch_contexts():
    global contexts
    contexts = cur.execute("SELECT * FROM contexts").fetchall()


def process_set_subcommands(client, msg, messages, subcommands, content):
    content_chunks = content.strip().split()
    command = content_chunks[0].lower()
    if command == "context":
        if PERMISSIONS_SET_CONTEXT == "admin" and not is_admin(client, msg):
            send_reply("Sorry, only admins can un/set contexts", msg)
            return

        context_name = content_chunks[1].lower()

        disabled_contexts = ["topic", "stream", "new", "help", "continue",
                             "contexts", "gpt3", "gpt4", "set", "unset", "me", "admin", "stats"]
        if context_name in disabled_contexts:
            send_reply(f"Sorry, you can't set context for {context_name}", msg)
            return

        context_value = " ".join(content_chunks[2:])
        upsert_context(context_name, context_value)
        send_reply(f"I have set !{context_name} to: {context_value}", msg)


def process_unset_subcommands(client, msg, messages, subcommands, content):
    content_chunks = content.strip().split()
    command = content_chunks[0].lower()
    if command == "context":
        if PERMISSIONS_SET_CONTEXT == "admin" and not is_admin(client, msg):
            send_reply("Sorry, only admins can un/set contexts", msg)
            return

        context_name = content_chunks[1].lower()
        delete_context(context_name)
        send_reply(f"I have unset !{context_name}", msg)


def handle_message(event):
    global contexts

    logging.debug("Handling event type: {type}".format(type=event['type']))

    if event['type'] != 'message':
        return

    msg = event['message']
    content = msg['content'].strip()

    if msg['sender_email'] == client.email:
        logging.debug("Ignoring message sent by myself")
        return

    if msg['type'] != 'private' and not re.search("@\*\*{bot}\*\*".format(bot=BOT_NAME), content) and not re.search("@{bot}".format(bot=BOT_NAME), content):
        logging.debug(
            "Ignoring message not mentioning the bot or sent in private")
        return

    # get subcommands (words starting with exclamation mark)
    subcommands = get_subcommands(content)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sender_email = msg['sender_email']
    logging.info("%s; %s; subcommands: %s; content: %s", str(current_time), str(sender_email), ",".join(subcommands), content)

    # first get rid of the command or mention trigger
    content = re.sub("@\*\*{bot}\*\*".format(bot=BOT_NAME), "", content)
    content = re.sub("@{bot}".format(bot=BOT_NAME), "", content)
    content = content.strip()
    content = remove_subcommands(content, subcommands)

    if subcommands and "help" in subcommands:
        print_help(msg)
        return

    model_tokens = {
        # input limit for GPT-3.5 Turbo (context 4k, prompt 2.5k, response 1.5k)
        'gpt-3.5-turbo': 3600,
        # input limit for GPT-4 (context 8k, prompt 6k, response 2k)
        'gpt-4': 6000,
        'gpt-4-0314': 6000,
        'gpt-4-0613': 6000,
        'gpt-4-1106-preview': 100000  # 128000
    }

    model = DEFAULT_MODEL_NAME or 'gpt-3.5-turbo'

    # available_models = ['gpt-3.5-turbo', 'gpt4']
    # TODO get default model from settings or check !settings

    if "gpt3" in subcommands:
        model = 'gpt-3.5-turbo'
    elif "gpt4" in subcommands:
        model = 'gpt-4-1106-preview'

    token_limit = model_tokens[model]

    messages = [
        {"role": "system", "content": os.environ['BOT_ROLE']},
        {"role": "user", "content": f"{content}"},
    ]

    context_names = [context[0] for context in contexts]
    context_map = {context[0]: context[1] for context in contexts}

    if "contexts" in subcommands:
        help_message = "Available contexts:\n"
        for context_name, context_value in contexts:
            help_message += f"- `!{context_name}`: {context_value}\n"
        send_reply(help_message, msg)
        return

    if "me" in subcommands:
        send_reply("This functionality is not implemented yet.", msg)
        return

    if "set" in subcommands:
        process_set_subcommands(client, msg, messages, subcommands, content)
        return

    if "unset" in subcommands:
        process_unset_subcommands(client, msg, messages, subcommands, content)
        return
    # new messages items will be appended after this index
    # as we add custom role: system messages here
    # and then add history messages later too between system and latest user message
    append_after_index = 1

    # iterate context_names and check if any of them is in subcommands
    for context_name in context_names:
        if context_name in subcommands:
            context_value = context_map[context_name]
            messages.insert(append_after_index, {
                            "role": "system", "content": f"{context_value}"})
            append_after_index += 1

    if "clear" in subcommands:
        send_reply(CONTEXT_CLEARED, msg)
        return
    
    if "continue" in subcommands:
        messages = with_previous_messages(client, msg, messages, subcommands, token_limit, append_after_index)

    logging.debug(messages)
    try:
        completion = openai_client.chat.completions.create(
            messages=messages,
            model=model,
            timeout=15
        )
        response = completion.choices[0].message.content
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        # return response, prompt_tokens, completion_tokens
        reply = f'{response}\n(tokens: prompt={prompt_tokens}, completion={completion_tokens})'
        logging.info(f'{current_time}; {sender_email}; prompt_tokens={prompt_tokens}; completion_tokens={completion_tokens}')

    except Exception as e:
        logging.error(e)
        reply = f"GPT API error: {e}"
        
    send_reply(reply, msg)


def main():
    global contexts
    logging.info("Initiate DB...")
    cur.execute("CREATE TABLE IF NOT EXISTS contexts(name PRIMARY KEY, value)")

    refetch_contexts()
    logging.info("Contexts")
    logging.info(contexts)

    result = client.get_profile()
    logging.debug(result)

    if (result.get('code') == 'UNAUTHORIZED'):
        logging.error("Invalid API key")
        sys.exit(1)

    logging.info("Starting the GPT Zulip bot named: {bot}".format(bot=BOT_NAME))
    client.call_on_each_event(handle_message, event_types=['message'])


if __name__ == "__main__":
    main()
