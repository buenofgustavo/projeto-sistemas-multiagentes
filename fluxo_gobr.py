"""Índices, acesso à API da OpenAI e interface de conversa para o notebook GoBR."""

import json
import os
from base64 import b64encode
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict
from uuid import uuid4
from zipfile import ZipFile

from langgraph.types import Command
from openai import OpenAI
from pydantic import ValidationError


MIME_ANEXOS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


def _extrair_indice(repo, destino, nome, padrao):
    with ZipFile(repo / f"{nome}.zip") as zip_indice:
        zip_indice.extractall(destino)
    manifestos = list(destino.glob(padrao))
    if len(manifestos) != 1:
        raise ValueError(f"Esperava um manifesto do índice em {nome}.zip.")
    manifesto = manifestos[0]
    return manifesto.parent, json.loads(manifesto.read_text(encoding="utf-8"))


def preparar_indices(repo):
    """Abre os ZIPs versionados no repositório e devolve os dois buscadores."""
    import torch
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    with TemporaryDirectory(prefix="gobr-indices-") as pasta_temporaria:
        destino = Path(pasta_temporaria)
        pasta_exemplos, exemplos = _extrair_indice(
            repo, destino, "explicacoes", "explicacoes/*/manifesto.json",
        )
        pasta_manual, manual = _extrair_indice(
            repo, destino, "manual", "manual/*/indice/manifesto.json",
        )
        if exemplos["modelo"] != manual["modelo_embedding"]:
            raise ValueError("Os índices precisam usar o mesmo modelo de embeddings.")

        gpu = torch.cuda.is_available()
        embedding = HuggingFaceEmbedding(
            model_name=exemplos["modelo"],
            device="cuda" if gpu else "cpu",
            max_length=max(exemplos["execucao"]["max_length"], manual["comprimento_modelo"]),
            model_kwargs={"torch_dtype": torch.float16} if gpu else {},
        )

        def abrir(pasta, limite):
            storage = StorageContext.from_defaults(persist_dir=str(pasta))
            indice = load_index_from_storage(storage, embed_model=embedding)
            return indice.as_retriever(similarity_top_k=limite)

        return abrir(pasta_exemplos, 2), abrir(pasta_manual, 3)


def _chave_openai():
    chave = os.getenv("OPENAI_API_KEY")
    if not chave:
        try:
            from google.colab import userdata
        except ImportError:
            raise RuntimeError("Defina a variável de ambiente OPENAI_API_KEY.") from None
        chave = userdata.get("OPENAI_API_KEY")
    if not chave:
        raise RuntimeError("Preencha OPENAI_API_KEY.")
    return chave


def preparar_recursos(repo=None, modelo="gpt-6-luna"):
    """Configura a API da OpenAI e abre os índices do checkout."""
    repo = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parent
    cliente = OpenAI(api_key=_chave_openai())
    exemplos, manual = preparar_indices(repo)
    return RecursosGoBR(repo, cliente, modelo, exemplos, manual)


class Estado(TypedDict, total=False):
    pergunta: str
    anexo: bytes
    mime_anexo: str
    leitura_anexo: str
    historico: list[dict[str, str]]
    contexto: str
    fontes: list[str]
    rascunho: str
    parecer: str
    mensagem: str
    codigo: str


class RecursosGoBR:
    def __init__(self, repo, cliente, modelo, exemplos, manual):
        self.repo = repo
        self.cliente = cliente
        self.modelo = modelo
        self.exemplos = exemplos
        self.manual = manual

    def buscar_contexto(self, estado: Estado):
        consulta = "\n".join(filter(None, [
            estado["pergunta"], estado["leitura_anexo"],
            *(h["resposta"] for h in estado["historico"]),
        ]))
        fontes, trechos = [], []

        if estado.get("mime_anexo") == "application/pdf":
            for pdf in sorted((self.repo / "dados/exemplos").glob("*.pdf")):
                regra = pdf.with_suffix(".gobr")
                if regra.is_file() and pdf.read_bytes() == estado["anexo"]:
                    fonte = regra.relative_to(self.repo).as_posix()
                    codigo = regra.read_text(encoding="utf-8-sig")
                    fontes.append(fonte)
                    trechos.append(f"[{fonte}] Exemplo GoBR correspondente ao anexo:\n{codigo}")
                    break

        for hit in self.exemplos.retrieve(consulta):
            dados = hit.node.metadata
            fonte = f"{dados['fonte_id']} ({dados['arquivo']})"
            codigo = (self.repo / dados["arquivo"]).read_text(encoding="utf-8-sig")
            fontes.append(fonte)
            trechos.append(f"[{fonte}]\n{hit.node.text}\n{codigo}")

        consultas_manual = [consulta]
        if estado["historico"]:
            consultas_manual.append(estado["historico"][-1]["resposta"])

        for consulta_manual in dict.fromkeys(consultas_manual):
            for hit in self.manual.retrieve(consulta_manual):
                dados = hit.node.metadata
                paginas = ", ".join(map(str, dados["paginas_pdf"]))
                fonte = f"{dados['chunk_id']} ({dados['fonte']}, páginas PDF {paginas})"
                if fonte not in fontes:
                    fontes.append(fonte)
                    trechos.append(f"[{fonte}]\n{hit.node.text}")

        return {"contexto": "\n\n".join(trechos), "fontes": fontes, "rascunho": "", "parecer": ""}

    def consultar(self, formato, instrucao, estado: Estado, mensagem):
        partes = [{"type": "input_text", "text": mensagem}]
        if "anexo" in estado:
            mime = estado["mime_anexo"]
            dados = b64encode(estado["anexo"]).decode("ascii")
            url_anexo = f"data:{mime};base64,{dados}"
            if mime == "application/pdf":
                partes.append({"type": "input_file", "file_data": url_anexo, "filename": "anexo.pdf"})
            else:
                partes.append({"type": "input_image", "image_url": url_anexo, "detail": "auto"})

        try:
            resposta = self.cliente.responses.parse(
                model=self.modelo,
                instructions=instrucao.strip(),
                input=[{"role": "user", "content": partes}],
                text_format=formato,
                max_output_tokens=8192,
                store=False,
            )
        except ValidationError:
            raise RuntimeError("A API da OpenAI devolveu uma resposta incompleta ou fora do formato esperado.") from None

        if resposta.status != "completed":
            motivo = resposta.incomplete_details.reason if resposta.incomplete_details else resposta.status
            raise RuntimeError(f"A API da OpenAI interrompeu a resposta ({motivo}).")

        decisao = resposta.output_parsed
        if decisao is None:
            raise RuntimeError("O modelo não devolveu uma resposta estruturada.")
        decisao.texto = decisao.texto.strip()
        if not decisao.texto:
            raise ValueError("O modelo devolveu uma resposta sem conteúdo.")
        return decisao


def contexto_do_pedido(estado: Estado):
    pedido = json.dumps({
        "pergunta": estado["pergunta"],
        "leitura_do_anexo": estado["leitura_anexo"],
        "esclarecimentos": estado["historico"],
    }, ensure_ascii=False)
    return f"Pedido e histórico:\n{pedido}\n\nReferências de sintaxe e estrutura:\n{estado['contexto']}"


def perguntar(agente, pergunta: str | None = None, arquivo: str | Path | None = None):
    """Exibe o resultado da conversa e retorna o GoBR aprovado na revisão."""
    pergunta = (pergunta or "").strip()
    if not pergunta and arquivo is None:
        raise ValueError("Informe uma pergunta ou anexe uma imagem ou um PDF.")

    estado = Estado(
        pergunta=pergunta or "Gere a parametrização GoBR a partir do anexo.",
        historico=[],
    )
    if arquivo is not None:
        caminho = Path(arquivo).expanduser()
        mime = MIME_ANEXOS.get(caminho.suffix.lower())
        if not mime:
            raise ValueError("Use um arquivo PNG, JPG, WebP ou PDF.")
        anexo = caminho.read_bytes()
        if not anexo:
            raise ValueError(f"O arquivo está vazio: {caminho}")
        estado.update(anexo=anexo, mime_anexo=mime)

    thread_id = uuid4().hex
    sessao = {"configurable": {"thread_id": thread_id}}
    try:
        resultado = agente.invoke(estado, sessao)
        while resultado.get("__interrupt__"):
            print(resultado["__interrupt__"][0].value)
            resposta = input("Sua resposta (Enter para encerrar): ").strip()
            if not resposta:
                return None
            resultado = agente.invoke(Command(resume=resposta), sessao)

        codigo = resultado.get("codigo")
        if codigo:
            print(codigo)
            print("\nReferências recuperadas")
            print("\n".join(resultado["fontes"]))
        else:
            print("Parametrização pendente.", resultado["mensagem"])
        return codigo
    finally:
        agente.checkpointer.delete_thread(thread_id)
