# agent_query.py
"""
Script final e único para consulta.
Implementa um agente de múltiplos passos que usa a base de conhecimento híbrida
para responder perguntas complexas de forma interativa.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import cohere
import argparse
from collections import defaultdict

# ==============================================================================
# 1. SETUP E CONFIGURAÇÃO
# ==============================================================================
load_dotenv()
COLL_NAME = os.getenv("UNIVERSAL_COLLECTION_NAME", "knowledge_base_main")

try:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY") or None)
    print("✅ Clientes das APIs inicializados.")
except Exception as e:
    print(f"❌ Erro ao inicializar clientes: {e}")
    exit()

EMBED_MODEL = "embed-multilingual-v3.0"
RERANK_MODEL = "rerank-multilingual-v3.0"
GEN_MODEL = "command-r-plus"

# ==============================================================================
# 2. FUNÇÕES DE BUSCA E GERAÇÃO (A "CAIXA DE FERRAMENTAS" DO AGENTE)
# ==============================================================================
def rerank_with_cohere(query: str, hits: list, top_k: int) -> list:
    if not hits:
        return []
    print(f"   -> Reordenando {len(hits)} candidatos para obter os {top_k} melhores...")
    docs_to_rerank = [hit.payload['full_text'] for hit in hits]
    try:
        rerank_results = co.rerank(model=RERANK_MODEL, query=query, documents=docs_to_rerank, top_n=top_k).results
        return [hits[result.index] for result in rerank_results]
    except Exception as e:
        print(f"   ⚠️ Erro no reranking: {e}. Retornando os top-k da busca inicial.")
        return hits[:top_k]

def vector_retrieval(query: str, target_document: str | None, initial_k: int, final_k: int) -> list:
    print(f"\n[BUSCA VETORIAL] Buscando por: '{query}'")
    query_filter = None
    if target_document:
        print(f"   -> Aplicando filtro para buscar APENAS no documento: '{target_document}'")
        query_filter = Filter(must=[FieldCondition(key="source_document", match=MatchValue(value=target_document))])
    else:
        print("   -> Buscando em TODOS os documentos da coleção.")

    query_vector = co.embed(texts=[query], model=EMBED_MODEL, input_type="search_query").embeddings[0]

    search_result = qdrant.query_points(
        collection_name=COLL_NAME,
        query=query_vector,
        query_filter=query_filter, 
        limit=initial_k
    )

    final_hits = rerank_with_cohere(query, search_result.points, top_k=final_k)
    print(f"   -> {len(final_hits)} trechos relevantes selecionados após reranking.")
    return final_hits

def graph_retrieval(node_ids: list[str]) -> str:
    print("\n[BUSCA NO GRAFO] Enriquecendo com contexto de relações...")
    if not node_ids:
        return "Nenhum nó do grafo foi encontrado nos documentos relevantes."

    print(f"   -> Buscando subgrafo para {len(node_ids)} nós...")
    query = "MATCH (n:Entity)-[r]-(m:Entity) WHERE n.id IN $ids RETURN n.name AS n1, type(r) AS rel, m.name AS n2 LIMIT 25"
    with neo4j_driver.session() as session:
        results = session.run(query, ids=node_ids)
        context = [f"({record['n1']})-[{record['rel']}]->({record['n2']})" for record in results]

    if not context:
        return "Nenhuma relação encontrada no grafo para os nós relevantes."

    context_str = "\n".join(context)
    print(f"   -> Contexto do grafo encontrado:\n{context_str}")
    return context_str

def hybrid_rag_tool(query: str, document_name: str | None = None) -> dict:
    print(f"\n🔎 Ferramenta RAG Híbrido chamada com a query: '{query}'")

    retrieved_docs = vector_retrieval(query, target_document=document_name, initial_k=25, final_k=5)
    if not retrieved_docs:
        return {"answer": "Não encontrei nenhum documento relevante para esta busca.", "follow_up_queries": [], "documents": [], "citations": []}

    neo4j_ids = set(id for doc in retrieved_docs for id in doc.payload.get('neo4j_ids', []))
    graph_context = graph_retrieval(list(neo4j_ids))

    documents_for_answer = [{"id": hit.id, "title": f"Trecho de '{hit.payload.get('source_document')}' ({hit.payload.get('type')})", "snippet": hit.payload['full_text']} for hit in retrieved_docs]

    answer_prompt = f"Com base nos documentos e no contexto do grafo, responda concisamente à pergunta do usuário. Pergunta: '{query}'.\n\nContexto do Grafo:\n{graph_context}"
    answer_response = co.chat(model=GEN_MODEL, message=answer_prompt, documents=documents_for_answer, temperature=0.1)

    context_for_pistas = "\n---\n".join([doc.payload['full_text'] for doc in retrieved_docs])
    pistas_prompt = f"""Analise o texto a seguir. Se ele mencionar explicitamente outras seções, documentos ou tópicos que seriam necessários para uma compreensão completa, liste-os como perguntas. Retorne uma lista JSON de strings. Se não houver referências claras, retorne uma lista vazia. Texto: "{context_for_pistas[:3000]}" """

    try:
        pistas_response = co.chat(model=GEN_MODEL, message=pistas_prompt, temperature=0.0)
        follow_up_queries = json.loads(pistas_response.text)
        if not isinstance(follow_up_queries, list):
            follow_up_queries = []
    except (json.JSONDecodeError, TypeError):
        follow_up_queries = []

    print(f" -> Resposta parcial da ferramenta: '{answer_response.text[:100].strip()}...'")
    print(f" -> Novas pistas para aprofundar: {follow_up_queries}")

    return {"answer": answer_response.text, "follow_up_queries": follow_up_queries, "citations": answer_response.citations, "documents": documents_for_answer}

# ==============================================================================
# 3. CONFIGURAÇÃO E EXECUÇÃO DO AGENTE
# ==============================================================================
tools = [
    {
        "name": "hybrid_rag_tool",
        "description": "Busca em uma base de conhecimento vetorial e de grafo para responder a uma pergunta. Retorna uma resposta, citações e sugestões de novas buscas para aprofundamento.",
        "parameter_definitions": {
            "query": {"description": "A pergunta a ser respondida ou o tópico a ser pesquisado.", "type": "str", "required": True},
            # ############# INÍCIO DA CORREÇÃO 1 #############
            "document_name": {
                "description": "Use este parâmetro SOMENTE SE o usuário pedir explicitamente para buscar em um arquivo ou documento específico. Se a pergunta for geral, NÃO preencha este parâmetro.",
                "type": "str",
                "required": False
            }
            # ############# FIM DA CORREÇÃO 1 #############
        }
    }
]
functions_map = {"hybrid_rag_tool": hybrid_rag_tool}

preamble = """Você é um assistente de pesquisa especialista e persistente. Sua tarefa é responder à pergunta do usuário da forma mais completa e precisa possível. Siga estes passos de forma metódica:
1. Comece usando a ferramenta `hybrid_rag_tool` com a pergunta original do usuário.
2. Analise a resposta e a lista de `follow_up_queries` retornada.
3. Se a resposta for completa e satisfatória, apresente-a como a resposta final. Não invente informações.
4. Se a resposta for incompleta OU se houver `follow_up_queries` relevantes que aprofundem o tópico, escolha a mais promissora e use a ferramenta `hybrid_rag_tool` novamente com essa nova query.
5. Continue este ciclo de busca e refinamento até ter confiança de que a pergunta original foi completamente respondida.
6. Ao final, sintetize todas as informações coletadas em uma resposta final coesa para o usuário.
"""

def run_agent(message: str, document_name: str | None = None):
    """Executa o agente de múltiplos passos para responder a uma consulta."""
    response = co.chat(model=GEN_MODEL, message=message, tools=tools, preamble=preamble, temperature=0.1)

    all_used_documents = {}
    all_citations = []

    while response.tool_calls:
        print("\n🤔 Ciclo de Raciocínio do Agente...")
        tool_results = []
        for tool_call in response.tool_calls:
            params = tool_call.parameters

            # ############# INÍCIO DA CORREÇÃO 2 #############
            # FORÇA o uso do document_name vindo do argumento --doc.
            # Se o usuário não passou --doc, document_name será None, e a busca será em todos os docs.
            # Isso previne que o agente invente um nome de documento.
            params['document_name'] = document_name
            # ############# FIM DA CORREÇÃO 2 #############

            print(f"   - Agente decidiu chamar a ferramenta: `{tool_call.name}` com os parâmetros: {params}")
            output = functions_map[tool_call.name](**params)

            if "documents" in output and output["documents"]:
                for doc in output["documents"]: all_used_documents[doc['id']] = doc
            if "citations" in output and output["citations"]:
                all_citations.extend(output["citations"])

            tool_results.append({"call": tool_call, "outputs": [output]})

        print("   - Agente está analisando os resultados para decidir o próximo passo...")

        response = co.chat(
            model=GEN_MODEL, message="", chat_history=response.chat_history,
            tools=tools, tool_results=tool_results, preamble=preamble, temperature=0.1
        )

    print("\n" + "="*25 + " RESPOSTA FINAL DO AGENTE " + "="*25)
    print(response.text)

    if all_citations:
        print("\n" + "-"*23 + " FONTES CITADAS " + "-"*23)
        doc_to_citations_map = defaultdict(list)
        for citation in all_citations:
            for doc_id in citation.document_ids:
                doc_to_citations_map[doc_id].append(citation.text)

        for i, (doc_id, cited_texts) in enumerate(doc_to_citations_map.items()):
            cited_doc = all_used_documents.get(doc_id)
            if cited_doc:
                print("\n" + "─" * 58)
                print(f"Fonte [{i+1}]: {cited_doc.get('title', 'N/A')}")
                print(f"Trechos Específicos Citados: {', '.join([f'\"{t}\"' for t in set(cited_texts)])}")
                print("\nContexto Completo da Fonte:")
                print("```text")
                indented_snippet = "\n".join(["  " + line for line in cited_doc.get('snippet', '').splitlines()])
                print(indented_snippet)
                print("```")
    print("\n" + "="*70)

def main():
    """Função principal para executar o agente de consulta."""
    parser = argparse.ArgumentParser(description="Consulta a base de conhecimento com um agente de múltiplos passos.")
    parser.add_argument("query", type=str, help="A pergunta que você quer fazer.")
    parser.add_argument("--doc", type=str, default=None, help="(Opcional) Nome do arquivo de origem para filtrar a busca (ex: 'teste.json').")
    args = parser.parse_args()

    # Se o usuário passou --doc, usamos o nome do arquivo sem a extensão. Se não, é None.
    target_doc_stem = Path(args.doc).stem if args.doc else None

    try:
        run_agent(message=args.query, document_name=target_doc_stem)
    finally:
        if neo4j_driver:
            neo4j_driver.close()
            print("\nConexão com o Neo4j fechada.")

if __name__ == "__main__":
    main()