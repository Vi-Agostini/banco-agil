"""Controles do campo de mensagem durante o processamento no Streamlit."""


def chat_input_controls(busy: bool) -> str:
    # Apenas o booleano da aplicação entra no script; nunca texto do cliente,
    # respostas do provedor ou credenciais.
    return """<script>
    (() => {
      const previous = window.bancoAgilComposer;
      if (previous) previous.dispose();
      const busy = __BUSY__;
      const scope = '.st-key-customer_message';
      const buttonSelector = scope + ' [data-testid="stChatInputSubmitButton"]';
      const inputSelector = scope + ' textarea';

      const syncButton = () => {
        const button = document.querySelector(buttonSelector);
        const input = document.querySelector(inputSelector);
        if (!button || !input) return;
        if (busy) {
          button.dataset.bankBusy = 'true';
          if (!button.disabled) button.disabled = true;
        } else if (button.dataset.bankBusy) {
          delete button.dataset.bankBusy;
          button.disabled = !input.value.trim();
        }
      };
      const blockSubmission = event => {
        if (!busy || !(event.target instanceof Element)) return;
        const enter = event.type === 'keydown' && event.key === 'Enter'
          && !event.shiftKey && !event.isComposing && event.target.matches(inputSelector);
        const click = event.type === 'click' && event.target.closest(buttonSelector);
        if (enter || click) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
      };
      // O React pode recriar o botão ou mudar o bloqueio ao editar o rascunho.
      // A captura bloqueia o envio antes de o observador atualizar o estado.
      const observer = new MutationObserver(syncButton);
      observer.observe(document.body, {subtree: true, childList: true,
        attributes: true, attributeFilter: ['disabled']});
      document.addEventListener('keydown', blockSubmission, true);
      document.addEventListener('click', blockSubmission, true);
      syncButton();
      window.bancoAgilComposer = {dispose: () => {
        observer.disconnect();
        document.removeEventListener('keydown', blockSubmission, true);
        document.removeEventListener('click', blockSubmission, true);
      }};
    })();
    </script>""".replace("__BUSY__", "true" if busy else "false")
