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
    template="""Edit this Answer by replacing meta-references with Corpus names.

Sources (with Corpus labels):
{context}

Answer to edit:
{answer}

Rules:
1. Find ALL phrases that refer to "where the information came from" in a meta way, such as:
   - "the sources", "provided sources", "referenced studies", "cited works"
   - "the documents", "the studies", "the provided context"
   - "based on X", "according to X", "mentioned in X" (where X = any meta-reference)

2. Replace those phrases with the actual Corpus name from Sources:
   - Use "Information for patients" or "Colonoscopy literature". Use both if content came from both corpora.
   - To decide which Corpus: check the Corpus labels in Sources above

3. Fix medical term spelling by copying exactly from Sources (e.g. if Sources says "diverticular", use "diverticular").

4. Keep sentence structure and all other words identical. Only replace meta-references with Corpus names.

5. Do NOT expand, summarize, or add new information.

Output the edited answer only:""",
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
