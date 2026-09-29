import pytest

from agentic_rag.agent_prompts import prompt_domain_guardrail, prompt_rewrite_medical, prompt_topic

pytestmark = pytest.mark.gpu


def _invoke(prompt, model, values, *, max_new_tokens: int) -> str:
    bound = model.bind(temperature=1.0, top_p=1.0, top_k=50, do_sample=False, max_new_tokens=max_new_tokens)
    messages = prompt.invoke(values).to_messages()
    return str(bound.invoke(messages)["content"]).strip()


def _rewrite(model, history: str, question: str) -> str:
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    bound = model.bind(temperature=0.1, top_p=0.95, top_k=50, do_sample=True, max_new_tokens=128)
    messages = prompt_rewrite_medical.invoke({"question": question, "history": history}).to_messages()
    return str(bound.invoke(messages)["content"]).strip()


@pytest.mark.parametrize(
    "question",
    [
        "Si vede anche lo stomaco?",
        "Cosa sono le lesioni piatte?",
        "Se ho ancora sangue nelle feci dopo un esame negativo cosa devo fare?",
        "Cos'è la marcatura fecale?",
        "Mi date un camice?",
        "Che gas viene usato?",
        "L'anidride carbonica è pericolosa?",
        "È come la risonanza magnetica?",
        "Posso parlare con il personale?",
        "È normale avere la bocca secca?",
        "È normale avere sonnolenza?",
        "È normale avere il cuore che batte più veloce?",
        "Cosa devo fare se perdo sangue dal retto?",
        "Quando devo andare in pronto soccorso?",
        "Ci sono rischi per il cuore?",
        "Le radiazioni sono pericolose?",
        "Ho sospetto di perforazione: posso farla?",
        "Posso tornare al lavoro?",
        "Quando devo preoccuparmi?",
        "Se mi viene dolore agli occhi dopo cosa devo fare?",
        "Devo depilarmi?",
        "Ho la creatinina alta, è un problema?",
        "Come espello il gas?",
        "Cosa sono i polipi?",
        "Quanto dura?",
        "Serve l'impegnativa?",
        "Devo essere a digiuno?",
        "Posso bere acqua prima?",
        "Devo sospendere gli anticoagulanti?",
        "Posso prendere la pillola per la pressione?",
        "È necessario che venga accompagnato?",
        "Posso farla se sono incinta?",
        "Il contrasto contiene iodio?",
        "La sonda rettale fa male?",
        "Quanto gonfiano l'intestino?",
        "In quale posizione mi mettono?",
        "Posso mangiare appena finito?",
        "Quando arriva il referto?",
        "Riesce a vedere i diverticoli?",
        "Se trovano qualcosa, fanno anche una biopsia?",
        "Può non riuscire bene se il colon non è pulito?",
        "Posso allenarmi la sera stessa?",
        "È normale avere crampi dopo?",
        "Chi chiamo se mi viene la febbre?",
        "A cosa servono i marcatori nelle feci?",
        "Il gas che usano può essere dannoso?",
        "Quali sintomi richiedono assistenza urgente?",
        "Dopo vedo sfocato: devo preoccuparmi?",
    ],
)
def test_domain_guardrail_accepts_observed_implicit_questions(gpu_guardrail_llm, question):
    result = _invoke(prompt_domain_guardrail, gpu_guardrail_llm, {"question": question}, max_new_tokens=5)

    assert result.upper() == "ON_TOPIC"


@pytest.mark.parametrize(
    "question",
    [
        "Come si prepara la carbonara?",
        "Chi ha vinto la partita?",
        "Come scrivo un programma in Python?",
        "Come posso curare il mal di gola?",
        "Qual è la capitale della Francia?",
    ],
)
def test_domain_guardrail_rejects_clearly_unrelated_questions(gpu_guardrail_llm, question):
    result = _invoke(prompt_domain_guardrail, gpu_guardrail_llm, {"question": question}, max_new_tokens=5)

    assert result.upper() == "OFF_TOPIC"


@pytest.mark.parametrize(
    ("history", "question", "expected_context"),
    [
        (
            "Q: Che gas viene usato durante la Colon-TC?\nA: Viene usata anidride carbonica.",
            "Come lo espello?",
            ("gas", "anidride carbonica", "co2"),
        ),
        (
            "Q: Cosa devo fare dopo la Colon-TC?\nA: Puoi riprendere gradualmente le normali attività.",
            "E dopo posso guidare?",
            ("colon-tc", "colon tc", "colonscopia"),
        ),
        (
            "Q: Durante la Colon-TC viene usato il contrasto?\nA: Dipende dal protocollo previsto.",
            "E se sono allergico?",
            ("contrasto",),
        ),
    ],
)
def test_rewritten_followup_remains_on_topic(gpu_guardrail_llm, history, question, expected_context):
    rewritten = _rewrite(gpu_guardrail_llm, history, question)
    result = _invoke(prompt_domain_guardrail, gpu_guardrail_llm, {"question": rewritten}, max_new_tokens=5)

    assert rewritten.casefold() != question.casefold()
    assert any(term in rewritten.casefold() for term in expected_context)
    assert result.upper() == "ON_TOPIC"


def test_unrelated_followup_is_new_and_off_topic(gpu_guardrail_llm):
    history = "Q: Come mi preparo alla Colon-TC?\nA: Segui la dieta e la preparazione prescritte."
    question = "Chi ha vinto la partita?"
    topic = _invoke(prompt_topic, gpu_guardrail_llm, {"question": question, "history": history}, max_new_tokens=5)
    domain = _invoke(prompt_domain_guardrail, gpu_guardrail_llm, {"question": question}, max_new_tokens=5)

    assert topic.upper() == "NUOVO"
    assert domain.upper() == "OFF_TOPIC"


def test_new_on_topic_followup_is_domain_accepted(gpu_guardrail_llm):
    history = "Q: Ho il glaucoma: posso ricevere il Buscopan?\nA: Devi segnalarlo al personale sanitario."
    question = "A che ora devo presentarmi per la Colon-TC?"
    topic = _invoke(prompt_topic, gpu_guardrail_llm, {"question": question, "history": history}, max_new_tokens=5)
    domain = _invoke(prompt_domain_guardrail, gpu_guardrail_llm, {"question": question}, max_new_tokens=5)

    assert topic.upper() == "NUOVO"
    assert domain.upper() == "ON_TOPIC"
