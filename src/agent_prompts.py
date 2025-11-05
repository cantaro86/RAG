# ------------------------
# Agent Prompts
# ------------------------


from langchain.prompts import PromptTemplate

prompt_validate_medical = PromptTemplate(
    template="""Classify the USER QUESTION.
    QUESTION:
    {question}

    Return "medical" if the question is about medicine, diseases, colonoscopy, polyps, CT colonography,
    biopsy, radiology, gastroenterology, etc.
    Otherwise return: "general".

    Respond ONLY with JSON: {{"score": "medical"}} or {{"score": "general"}}.
    """,
    input_variables=["question"],
)


prompt_general = PromptTemplate(
    template="""You are a scientific assistant. Answer the question accurately and concisely.

    Conversation history (most recent first):
    {history}

    Current question:
    {question}
    """,
    input_variables=["history", "question"],
)


prompt_rag = PromptTemplate(
    template="""You are a scientific assistant. Use the context below to answer the question accurately and concisely.

    Conversation history (most recent first):
    {history}

    Retrieved context:
    {context}

    Current question:
    {question}
    """,
    input_variables=["history", "context", "question"],
)


# Hallucination check
prompt_hallucination = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in the retrieved documents.
    Here are the documents:
    {documents}
    Here is the answer: {generation}
    Respond ONLY with JSON: {{"score": "yes"}} or {{"score": "no"}}.""",
    input_variables=["generation", "documents"],
)


# Usefulness check
prompt_usefulness = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to the user question.
    Question: {question}
    Answer: {generation}
    Respond ONLY with JSON: {{"score": "yes"}} or {{"score": "no"}}.""",
    input_variables=["generation", "question"],
)


# Question rewriter
prompt_rewrite_medical = PromptTemplate(
    template="""You are a professional assistant specialized in reformulating medical questions
to improve information retrieval.

Your task:
- Given the conversation history and the user's last question, rewrite the question so that it is fully explicit
and unambiguous.
- Replace pronouns like "their", "they", "it", "this", etc. with the actual referenced entity from the history.
- Rephrase the question **only if necessary** to make it clearer and more likely to match relevant documents.
- Do not change its meaning, specificity, or focus. Keep the topic identical.
- Produce only the question, no introductions, explanations, lists, or multiple options.

Question: {question}

Conversation history (oldest first): {history}

""",
    input_variables=["question", "history"],
)
