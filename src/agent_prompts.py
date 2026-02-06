# ------------------------
# Agent Prompts
# ------------------------


from langchain_core.prompts import PromptTemplate

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
prompt_clean = PromptTemplate(
    template="""Edit this Answer. Make ONLY minimal changes.

Sources:
{context}

Answer:
{answer}

Rules:
1. Remove ALL these phrases:
   "according to the sources", "based on the sources", "provided sources",
   "given sources", "the sources", "Source [N]", "the context", "retrieved context",
   "conversation history”, “based on the documents”, or similar meta-talk.

2. Replace them with the content of Corpus labels of the Sources used, choosing among:
   "Information for patients"
   "Colonoscopy literature" or both.

3. Copy medical term spellings exactly from Sources (e.g., if Sources says "diverticular", use "diverticular").

4. Do NOT add, rewrite, expand, or summarize.

Output ONLY the cleaned answer (no explanations, no quotes).""",
    input_variables=["context", "answer"],
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
