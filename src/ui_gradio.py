# src/ui_gradio.py
import uuid

import gradio as gr

from src._load_env import Config
from src.agent_factory import build_rag_agent
from src.bilingual_question import BilingualQuestion
from src.loggers import Logger

logger = Logger.get_logger(__name__)


def launch_gradio(cfg: Config):
    agent = build_rag_agent(cfg)

    def respond(message: str, history, thread_id: str):
        if not history:
            thread_id = str(uuid.uuid4())

        try:
            quest = BilingualQuestion(message)
            logger.debug(f"Language = {quest.lang}, class = {quest}")
        except ValueError as e:
            logger.error(f"Error processing question: {e}")

        config = {"configurable": {"thread_id": thread_id}}
        agent_input = {"question": quest.en}

        last_output = None
        for output in agent.stream(agent_input, config=config):
            for _key, value in output.items():
                last_output = value

        if last_output and isinstance(last_output, dict) and "generation" in last_output:
            answer = last_output["generation"]
            if quest.lang == "it":
                answer_it = quest.translate_to_italian(answer)
                logger.info(f"Final answer (IT): {answer_it}")
            return answer_it, thread_id

        return "No generation returned from agent.", thread_id

    thread_state = gr.State(value=str(uuid.uuid4()))

    demo = gr.ChatInterface(
        fn=respond,
        additional_inputs=[thread_state],
        additional_outputs=[thread_state],
        title="RAG Agent",
    )

    demo.queue().launch(
        server_name=getattr(cfg, "gradio_host", "0.0.0.0"),
        server_port=int(getattr(cfg, "gradio_port", 7860)),
        share=getattr(cfg, "gradio_share", False),
    )
