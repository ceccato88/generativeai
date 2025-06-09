# ingest.py
"""
Lê um JSON estruturado e o ingere em uma coleção universal no Qdrant e Neo4j,
criando um vínculo explícito entre os vetores de chunks e os nós do grafo.
"""
import os
import uuid
import time
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import cohere
import argparse

# ==============================================================================
# 1. SETUP E CONFIGURAÇÃO
# ==============================================================================
load_dotenv()

# --- Clientes de API ---
try:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    neo4j = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_KEY") or None)
    print("✅ Clientes das APIs inicializados.")
except Exception as e:
    print(f"❌ Erro ao inicializar clientes: {e}")
    exit()

# --- Configurações de Ingestão ---
COLL_NAME = os.getenv("UNIVERSAL_COLLECTION_NAME", "knowledge_base_main")
EMBED_MODEL = "embed-multilingual-v3.0"
GEN_MODEL = "command-r-plus"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1024))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", 64))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))

# ==============================================================================
# 2. MODELOS Pydantic E FUNÇÕES AUXILIARES
# ==============================================================================
class SingleEdge(BaseModel):
    node: str | None = None
    target_node: str | None = None
    relationship: str | None = None

class GraphPayload(BaseModel):
    graph: list[SingleEdge] = Field(default_factory=list)


def extract_graph_from_text(text: str) -> GraphPayload | None:
    prompt = f"""Sua tarefa é extrair um grafo de conhecimento do texto fornecido. Identifique as principais entidades (nós) e suas relações (arestas). Sua resposta DEVE ser um objeto JSON válido e nada mais, começando com {{ e terminando com }}. O JSON deve ter uma única chave "graph", contendo uma lista de objetos com "node", "target_node" e "relationship". Texto para análise:\n```{text}```"""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = co.chat(model=GEN_MODEL, message=prompt, temperature=0.0)
            raw = resp.text.strip()
            json_start = raw.find('{')
            json_end = raw.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return GraphPayload.model_validate_json(raw[json_start:json_end])
            raise ValueError("Nenhum objeto JSON encontrado na resposta do modelo.")
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            if attempt == LLM_MAX_RETRIES:
                print(f"   └─ ⚠️  Falha final na extração: {e}")
                return None
            time.sleep(1.5 ** attempt)
    return None

def format_table_for_llm(el: dict) -> str:
    title = el.get('title', 'Tabela')
    cells = el.get('cells', {})
    if not cells:
        return title
    try:
        max_row = max(int(k.split('_')[0]) for k in cells)
        max_col = max(int(k.split('_')[1]) for k in cells)
    except (ValueError, IndexError):
        return title
    header = [cells.get(f'0_{c}', {}).get('text', '').strip() for c in range(max_col + 1)]
    rows = ["| " + " | ".join([cells.get(f'{r}_{c}', {}).get('text', '').strip() for c in range(max_col + 1)]) + " |" for r in range(1, max_row + 1)]
    return f"**{title}**\n\n| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n" + "\n".join(rows)

def generate_summaries(items: list[dict]) -> None:
    print("\n[ETAPA 4/5] Gerando resumos para busca semântica...")
    prompt_template = "Resuma o seguinte trecho ({type}) de um documento técnico de forma concisa para uma busca semântica:\n\n```{element}```"
    for i, item in enumerate(items):
        print(f"   -> Gerando resumo para o item {i+1}/{len(items)}...", end='\r')
        prompt = prompt_template.format(type=item['type'], element=item['full_text'])
        try:
            item['summary'] = co.chat(model=GEN_MODEL, message=prompt, temperature=0.1).text
        except Exception as e:
            print(f"   └─ ⚠️ Erro ao sumarizar item {i+1}: {e}")
            item['summary'] = item['full_text'][:250]
    print("\n✅ Resumos gerados.")

# ==============================================================================
# 3. FUNÇÕES DE ORQUESTRAÇÃO (REFATORADAS)
# ==============================================================================

def load_and_split_document(json_path: Path, source_document_name: str) -> list[dict]:
    """Carrega o JSON e o divide em chunks processáveis (texto, código, tabelas)."""
    print("\n[ETAPA 1/5] Carregando e dividindo o documento...")
    with open(json_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)['data']

    generic_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    processed_items = []
    for el in doc_data.get('elements', []):
        item_type, item_text = el.get('type'), el.get('text', '')
        if not item_text:
            continue

        item_base = {"source_document": source_document_name}
        if item_type == 'paragraph':
            is_code = "```python" in item_text
            splitter = python_splitter if is_code else generic_splitter
            for chunk in splitter.split_text(item_text):
                processed_items.append({**item_base, "id": str(uuid.uuid4()), "type": 'code' if is_code else 'text', "full_text": chunk})
        elif item_type == 'table':
            processed_items.append({**item_base, "id": str(uuid.uuid4()), "type": 'table', "full_text": format_table_for_llm(el)})

    print(f"   -> Documento dividido em {len(processed_items)} itens.")
    return processed_items

def build_knowledge_graph(processed_items: list[dict]) -> tuple[dict, list]:
    """Constrói o grafo de conhecimento extraindo entidades de cada chunk."""
    print("\n[ETAPA 2/5] Extraindo grafo de conhecimento de cada chunk...")
    nodes, rels = {}, []
    for i, item in enumerate(processed_items):
        print(f"   -> Extraindo grafo do item {i+1}/{len(processed_items)}...", end='\r')
        graph_payload = extract_graph_from_text(item['full_text'])
        if not (graph_payload and graph_payload.graph):
            continue

        for edge in graph_payload.graph:
            if edge.node and edge.target_node and edge.relationship:
                a = edge.node.strip()
                b = edge.target_node.strip()
                rel = edge.relationship.strip().upper().replace(" ", "_")
                if a and b and rel:
                    if a not in nodes: nodes[a] = str(uuid.uuid4())
                    if b not in nodes: nodes[b] = str(uuid.uuid4())
                    rels.append({"source": nodes[a], "target": nodes[b], "type": rel})

    print(f"\n   -> Grafo final construído com {len(nodes)} nós e {len(rels)} relações.")
    return nodes, rels

def link_chunks_to_nodes(processed_items: list[dict], nodes: dict) -> None:
    """Vincula chunks aos nós do grafo usando correspondência de palavra inteira (regex)."""
    print("\n[ETAPA 3/5] Vinculando chunks com nós do grafo...")
    for item in processed_items:
        item_nodes = []
        for name, node_id in nodes.items():
            if re.search(r'\b' + re.escape(name) + r'\b', item['full_text'], re.IGNORECASE):
                item_nodes.append(node_id)
        item['neo4j_ids'] = list(set(item_nodes))
    print("   -> Vínculos criados com sucesso.")

def ingest_data_stores(processed_items: list[dict], nodes: dict, rels: list, source_document_name: str) -> None:
    """Ingere os dados processados no Neo4j e Qdrant."""
    print("\n[ETAPA 5/5] Ingerindo dados no Neo4j e Qdrant...")

    if nodes:
        print("   -> Ingerindo no Neo4j...")
        with neo4j.session() as s:
            s.run("MATCH (n {source_document:$f}) DETACH DELETE n", f=source_document_name)
            for name, node_id in nodes.items():
                s.run("MERGE (e:Entity {id:$id}) SET e.name=$n, e.source_document=$f", id=node_id, n=name, f=source_document_name)
            for r in rels:
                s.run("MATCH (a:Entity {id:$s}), (b:Entity {id:$t}) MERGE (a)-[rel:RELATIONSHIP {type:$ty, source_document:$f}]->(b)", s=r["source"], t=r["target"], ty=r["type"], f=source_document_name)
        print("   ✅ Neo4j atualizado.")
    else:
        print("   -> Nenhum grafo para ingerir no Neo4j.")

    try:
        qdrant.get_collection(collection_name=COLL_NAME)
    except Exception:
        print(f"   -> Coleção '{COLL_NAME}' não encontrada. Criando...")
        dim = len(co.embed(texts=["."], model=EMBED_MODEL, input_type="search_document").embeddings[0])
        qdrant.create_collection(COLL_NAME, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        qdrant.create_payload_index(COLL_NAME, field_name="source_document", field_schema=models.PayloadSchemaType.KEYWORD, wait=True)

    print(f"   -> Deletando entradas antigas para '{source_document_name}' no Qdrant...")
    qdrant.delete(COLL_NAME, points_selector=Filter(must=[FieldCondition(key="source_document", match=MatchValue(value=source_document_name))]), wait=True)

    summaries = [item['summary'] for item in processed_items]
    embeds = co.embed(texts=summaries, model=EMBED_MODEL, input_type="search_document").embeddings

    # ############# INÍCIO DA CORREÇÃO #############
    # A variável correta é 'processed_items', não 'items'.
    points = [models.PointStruct(id=item['id'], vector=embeds[i], payload=item) for i, item in enumerate(processed_items)]
    # ############# FIM DA CORREÇÃO #############

    total_batches = (len(points) + QDRANT_BATCH_SIZE - 1) // QDRANT_BATCH_SIZE
    for i in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[i:i+QDRANT_BATCH_SIZE]
        print(f"   -> Enviando lote {i//QDRANT_BATCH_SIZE + 1}/{total_batches} para o Qdrant...")
        qdrant.upsert(collection_name=COLL_NAME, points=batch, wait=True)
    print("   ✅ Qdrant atualizado.")

# ==============================================================================
# 4. FLUXO PRINCIPAL DE INGESTÃO (ORQUESTRADOR)
# ==============================================================================
def main():
    """Orquestra todo o processo de ingestão."""
    parser = argparse.ArgumentParser(description="Ingere um documento JSON para a base de conhecimento híbrida (Grafo+Vetor).")
    parser.add_argument("json_path", type=str, help="Caminho para o arquivo .json a ser ingerido.")
    args = parser.parse_args()

    json_file_path = Path(args.json_path)
    if not json_file_path.exists():
        print(f"❌ Arquivo '{json_file_path}' não encontrado.")
        exit()

    source_document_name = json_file_path.stem
    print(f"\n>> Iniciando ingestão de '{source_document_name}' para a coleção '{COLL_NAME}'")
    start_time = time.time()

    processed_items = load_and_split_document(json_file_path, source_document_name)
    nodes, rels = build_knowledge_graph(processed_items)
    link_chunks_to_nodes(processed_items, nodes)
    generate_summaries(processed_items)
    ingest_data_stores(processed_items, nodes, rels, source_document_name)

    if neo4j:
        neo4j.close()
    print(f"\n✅ Processo concluído em {time.time() - start_time:.2f} segundos.")


if __name__ == "__main__":
    main()