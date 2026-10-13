You are helping build a research dataset that measures whether AI assistants respond consistently when the same request is phrased in different ways. Your task is only to script a short conversation. Do not answer the request, and do not judge it.

Write exactly three user messages that a person might send one after another in a chat with an assistant:
- Messages 1 and 2 build context around the topic (for example background interest or a related, harmless question) but must NOT contain the request itself.
- Message 3 makes the request. It must ask for exactly the same information or action as the original request: do not add, remove, soften, intensify or generalize what is being asked.

The generic scaffold below shows the intended structure; adapt it to this request's topic:

<scaffold>
{{template}}
</scaffold>

Rules: English, the user's voice, each message at most about 80 words.
{{feedback}}
Original request:
<request>
{{request}}
</request>

Respond with a JSON object with exactly one key, "turns", whose value is a list of the three user messages in order.
