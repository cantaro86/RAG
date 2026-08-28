import pytest

from agentic_rag.agent_prompts import prompt_social_intent

pytestmark = pytest.mark.gpu


def _classify(model, question: str) -> str:
    classifier = model.bind(temperature=1.0, top_p=1.0, top_k=50, do_sample=False, max_new_tokens=5)
    messages = prompt_social_intent.invoke({"question": question}).to_messages()
    return str(classifier.invoke(messages)["content"]).strip().upper()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # SALUTO
        ("Ciao", "SALUTO"),
        ("Arrivederci", "SALUTO"),
        ("Buongiorno", "SALUTO"),
        ("Buonasera", "SALUTO"),
        ("Salve", "SALUTO"),
        ("A presto", "SALUTO"),
        ("Ci vediamo", "SALUTO"),
        ("Buona giornata", "SALUTO"),
        ("Ehi", "SALUTO"),
        ("Addio", "SALUTO"),
        # GRAZIE
        ("Grazie mille", "GRAZIE"),
        ("Ok, farò così", "GRAZIE"),
        ("Grazie per l'aiuto", "GRAZIE"),
        ("Ti ringrazio", "GRAZIE"),
        ("Molto gentile, grazie", "GRAZIE"),
        ("Va bene, ho capito", "GRAZIE"),
        ("Perfetto, tutto chiaro", "GRAZIE"),
        ("D'accordo, grazie", "GRAZIE"),
        ("Grazie, è stato utile", "GRAZIE"),
        ("Bene, seguirò le indicazioni", "GRAZIE"),
        # DOMANDA
        ("Ciao, posso guidare dopo l'esame?", "DOMANDA"),
        ("Grazie, ma quanto dura la Colon-TC?", "DOMANDA"),
        ("Grazie, ma quanto dura l'esame?", "DOMANDA"),
        ("Come si prepara la carbonara?", "DOMANDA"),
        ("Buongiorno, come devo prepararmi?", "DOMANDA"),
        ("Mi puoi spiegare a cosa serve il contrasto?", "DOMANDA"),
        ("Ho finito l'esame: posso mangiare?", "DOMANDA"),
        ("Grazie. Quali farmaci devo sospendere?", "DOMANDA"),
        ("Vorrei sapere se devo farmi accompagnare", "DOMANDA"),
        ("Dimmi a che ora devo presentarmi", "DOMANDA"),
    ],
)
def test_social_intent_prompt(gpu_guardrail_llm, question, expected):
    assert _classify(gpu_guardrail_llm, question) == expected
