"""Processamento em segundo plano com trava por sessão e proteção contra reruns."""
from threading import Lock, Thread
from concurrent.futures import Future
from time import sleep as sleep_seconds

REPLY_DELAY_SECONDS = 5.0

BUSY = "Aguarde um momento, estou trabalhando nisso."


class ProcessingGuard:
    def __init__(self):
        self.lock = Lock()

    def __deepcopy__(self, memo):
        return type(self)()


class ProcessingJob:
    def __init__(self, orchestrator, session, prompt):
        self.session = session
        self._transcript_before = list(session.transcript)
        self.future = Future()
        self.thread = Thread(target=self._run, args=(orchestrator, session, prompt), daemon=True)
        self.thread.start()

    @property
    def visible_transcript(self):
        # Só revela a resposta após a pausa de apresentação, inclusive em reruns.
        new_messages = self.session.transcript[len(self._transcript_before):]
        return self._transcript_before + [item for item in new_messages if item[0] == "user"]

    def _wait_before_reply(self):
        sleep_seconds(REPLY_DELAY_SECONDS)

    def _run(self, orchestrator, session, prompt):
        try:
            self.future.set_result(orchestrator.handle(session, prompt, before_reply=self._wait_before_reply))
        except Exception:
            # Uma falha inesperada não deve repetir um turno parcialmente executado.
            text = "Há uma indisponibilidade técnica no momento. Tente novamente mais tarde."
            orchestrator.reply_recorder.say(session, text)
            session.transcript.append(("assistant", text))
            self.future.set_result(text)
