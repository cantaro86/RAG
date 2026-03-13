# ------------------------
# Agent Prompts
# ------------------------


from langchain_core.prompts import ChatPromptTemplate

prompt_topic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a topic continuity classifier for a medical Q&A system.

Task: Determine if the new question continues the SAME topic as the conversation history or introduces a NEW topic.

Definition:
- SAME TOPIC: the new question is about
- NEW TOPIC: the new question is about something else completely.

SAME topic means:
- Asking about the same entity, thing, medical concept, subject that was discussed.
- Follow-up questions like "what if...", "and what about...", "how about..."
- Asking about opposite/alternative conditions related to the same topic.
- Clarifications, variations, or additional details about the previous subject

NEW topic means:
- Completely different medical procedure, exam, or body system
- Unrelated condition or medication
- Question that has no connection to previous discussion

Respond with EXACTLY one of these two strings:
"SAME" or "NEW"
Do NOT add introductions, explanations, lists, or multiple options. """,
        ),
        (
            "user",
            """
Conversation History (oldest first):
{history}

New question:
{question}
""",
        ),
    ]
)


prompt_rag = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a medical assistant with clinical reasoning skills.

Hard rules:
1. Use ONLY the Sources below to generate your Answer.
2. LOGICAL CONSISTENCY: Verify that the condition described in the Sources matches the user's
   condition exactly.
   - Do not provide instructions for a condition the user has explicitly denied having.
3. For tables and sections:
   - If the Sources include the actual content of a table or section, use that content fully
     in your answer.
   - If the Sources only mention a table or section by name without including its rows or
     details, reference it by name and Corpus only. DO NOT invent or substitute its contents.
4. If NO source contains ANY part of the answer, output EXACTLY:
   "The Information for patients and the colonoscopy literature do not contain the information."
   DO NOT add disclaimers about individual sources that did not contribute to the answer.
   If a source is not relevant to the question, simply ignore it — do not mention its absence.
5. Do not refer to generic "sources" or "documents". Always specify the source by name using
   the "Corpus" field.
""",
        ),
        (
            "user",
            """Current question:
{question}

Sources:
{context}

Answer:""",
        ),
    ]
)


# Answer cleaner
prompt_clean_chat = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise medical editor.
Task: Copy the text below, removing ONLY meta-commentary about the retrieval process.

Rules:
1. REMOVE meta-commentary phrases about WHERE the assistant got the information:
   - "Based on the provided sources/context/documents..."
   - "According to the sources..."
   - "As mentioned in the conversation history..."
   - "The sources indicate..."
   - "As stated in the retrieved documents..."
2. KEEP references to the "Information for patients" and "colonoscopy literature".
3. REMOVE citation markers: [1], [Source 1], (Doc 2)
4. If you remove a prefix, capitalize the new start of the sentence.
5. Keep ALL other sentences exactly as they are.
6. DO NOT rephrase, summarize or expand.
7. Output ONLY the cleaned text (no preambles, no explanations, no examples)
8. DO NOT include the words "Text to edit" or similar markers
9. If the input text ends abruptly, DO NOT add any trailing text to the output.
10. DO NOT supply missing details.
""",
        ),
        (
            "user",
            """{answer}""",
        ),
    ]
)


# Question rewriter
prompt_rewrite_medical = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query rewriter for a medical information retrieval system.

Task: Given a Question and a Conversation history,
rewrite the question as a **standalone question** that can be understood without the conversation history.

Instructions:
1. Replace ALL implicit references with explicit entities from history:
   - Pronouns: "it", "this", "that", "these", "those", "they", "them", "their"
   - Implicit contrasts and negations: "what if not", "what if I don't", "instead",
     "what about", "and if", "how about", "rather than"
   - Implicit topic continuations: "what else", "anything else", "what about the other"
   - Ambiguous named entities: table numbers, section names, summary numbers, or any label,
    always qualify them with the corpus and context mentioned in history.
2. The history may span multiple turns — scan ALL of it to identify the relevant medical
   procedure, condition, and context, not just the most recent exchange.
3. If the question introduces a completely new topic unrelated to history, return it unchanged.
4. Do not answer the question or add new medical information.
5. Output ONLY the rewritten question, no introductions, explanations, lists, or multiple options.

""",
        ),
        (
            "user",
            """
            Question: {question}

            Conversation history: {history}
            """,
        ),
    ]
)


prompt_transform_query = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ You are a query rewriter for a medical information retrieval system.

Task: The question below failed to retrieve relevant documents.
Rephrase it to improve retrieval, without changing its meaning.

Instructions:
1. The question is already self-contained — do NOT add history context.
2. Expand or vary medical terminology: use synonyms, related clinical terms,
   or alternative phrasings that might match document language better.
3. If the question is composite, focus on its most specific retrievable aspect.
4. Do not answer the question or add new medical information.
5. Output ONLY the rewritten question.
""",
        ),
        (
            "user",
            """
            Question: {question}

            Conversation history: {history}
            """,
        ),
    ]
)
