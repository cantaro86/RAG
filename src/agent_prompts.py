# ------------------------
# Agent Prompts
# ------------------------


from langchain.prompts import PromptTemplate

prompt_rag = PromptTemplate(
    template="""You are a scientific assistant. Use the context below to answer the question accurately and concisely.

    Context:
    {context}

    Question:
    {question}

    Answer:""",
    input_variables=["context", "question"],
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
    template="""You are a professional assistant specialized in reformulating factual and medical questions
to improve information retrieval.

# Your task:
# - Rephrase the question **only if necessary** to make it clearer and more likely to match relevant documents.
# - If necessary, se standard medical terminology.
# - Do not change its meaning, specificity, or focus. Keep the topic identical.
# - Produce exactly ONE improved version of the question.
# - Do not generate explanations, lists, or multiple options.

Question: {question}

Answer:
""",
    input_variables=["question"],
)
