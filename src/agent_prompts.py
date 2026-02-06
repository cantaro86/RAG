# ------------------------
# Agent Prompts
# ------------------------


from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

prompt_topic = PromptTemplate(
    template="""You are a topic continuity classifier.

You must decide ONLY ONE THING:

Does the user's NEW QUESTION refer to the SAME topic as the previous conversation?

Definition:
- SAME TOPIC: the new question is about the same entity, thing, medical concept, subject that was discussed.
- NEW TOPIC: the new question is about something else completely.

Respond with EXACTLY one of these two strings:
"SAME" or "NEW"
Do NOT add introductions, explanations, lists, or multiple options.

Conversation History (oldest first):
{history}

New question:
{question}
""",
    input_variables=["history", "question"],
)


prompt_rag = PromptTemplate(
    template="""You are a medical assistant.

Hard rules:
- Use ONLY the information in Sources and Conversation history. Do not use outside knowledge.
- If Sources do not contain the answer, output EXACTLY:
  the exam documentation and the colonoscopy literature do not contain the information.

Conversation history:
{history}

Sources:
{context}

Current question:
{question}

Answer:""",
    input_variables=["history", "context", "question"],
)


# Answer cleaner
prompt_clean_chat = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a silent text editor.\n"
            "Output ONLY the edited answer text (no preambles, no explanations, no lists, no headings).\n"
            "Do not paraphrase or rephrase; make the smallest possible edits.\n"
            "Never output the phrase 'conversation history'.\n"
            "Do not add any new lines. Keep the same number of lines as the original.",
        ),
        (
            "user",
            """Answer:
    {answer}

    Sources (spelling only):
    {context}

    Do ONLY:
    1) Delete meta-reference phrases anywhere (e.g., 'Based on...', 'According to...', 'As stated in...',
    references to sources/documents/context/research, and 'Source [N]').
    2) Fix medical spelling ONLY when the correct spelling appears verbatim in Sources;
    otherwise do not change the term.

    Return ONLY the edited answer text.""",
        ),
    ]
)


# Question rewriter
prompt_rewrite_medical = PromptTemplate(
    template="""You are a professional assistant specialized in reformulating medical questions
to improve information retrieval.

Your task:
- **If the last question introduces a new topic unrelated to the history, do not rewrite it. Return it unchanged.**
- Given the conversation history and the user's last question, rewrite the question so that it is fully explicit
and unambiguous.
- Replace pronouns like "their", "they", "it", "this", etc. with the actual referenced entity from the history.
Do not add new information.
- Rephrase the question **only if necessary** to make it clearer and more likely to match relevant documents.
- Do not change its meaning, specificity, or focus. Keep the topic identical.
- **Keep the question short and concise.**
- Produce only the question, no introductions, explanations, lists, or multiple options.

Question: {question}

Conversation history (oldest first): {history}
""",
    input_variables=["question", "history"],
)
