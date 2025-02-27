Este notebook implementa uma infraestrutura abrangente para análise de dados experimentais, combinando técnicas de descoberta de estrutura causal, importância de características, visualização avançada e análise estatística. O foco principal está no processamento e análise de dados relacionados a erros de usuários e dificuldade de tarefas, explorando diferentes grupos experimentais.

## Funcionalidades Principais

- **Descoberta de Estrutura Causal**: Implementação do algoritmo FCI (Fast Causal Inference) para identificar relações causais entre variáveis.
- **Análise de Importância de Características**: Utilização de valores SHAP (SHapley Additive exPlanations) para quantificar a influência de cada característica nos erros observados.
- **Visualizações Avançadas**: Conjunto diversificado de gráficos incluindo KDE, violin plots, radar plots e grafos bipartidos.
- **Integração com LLMs**: Análise automatizada de insights através de modelos de linguagem como Mistral-7B.
- **Aprendizado por Reforço**: Implementação de um agente DDQN (Double Deep Q-Network) para aprendizado a partir dos dados experimentais.
- **Fine-tuning de LLMs**: Configuração para ajuste fino de modelos de linguagem utilizando a biblioteca Unsloth.
- **Quantização GGUF**: Pipeline para conversão e quantização de modelos para formato GGUF compatível com llama.cpp.

## Requisitos

```python
!pip install unsloth causal-learn shap transformers plotly accelerate bitsandbytes
```

Dependências principais:
- pandas
- matplotlib
- seaborn
- networkx
- shap
- causal-learn
- transformers
- unsloth (para fine-tuning eficiente de LLMs)
- pytorch
- plotly

## Estrutura do Código

### Configuração e Importações
O notebook começa com a verificação da disponibilidade de GPU para funcionalidades de LLM e configuração do ambiente Google Colab.

### Conjunto de Dados
Utiliza um conjunto de dados sintético com as seguintes colunas:
- `participant_id`: Identificador único do participante
- `group`: Grupo experimental (Group 1, Group 2, Group 3)
- `errors`: Número de erros cometidos
- `task_difficulty`: Nível de dificuldade da tarefa
- `error_type_1`, `error_type_2`, `error_type_3`: Diferentes tipos de erros
- `task_feature_1`, `task_feature_2`, `task_feature_3`: Características das tarefas

### Principais Componentes

#### 1. Pré-processamento de Dados
```python
def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the data by performing one-hot encoding on the 'group' column
    and scaling the numerical columns. Returns both the transformed DataFrame
    and a copy of the original DataFrame.
    """
    df_original = df.copy()  # Store a copy of the original DataFrame
    df = pd.get_dummies(df, columns=[GROUP_COLUMN], prefix=GROUP_COLUMN)
    encoded_group_cols = [col for col in df.columns if col.startswith(f"{GROUP_COLUMN}_")]
    df = scale_data(
        df,
        [
            ERRORS_COLUMN,
            TASK_DIFFICULTY_COLUMN,
            ERROR_TYPE_1_COLUMN,
            ERROR_TYPE_2_COLUMN,
            ERROR_TYPE_3_COLUMN,
            TASK_FEATURE_1_COLUMN,
            TASK_FEATURE_2_COLUMN,
            TASK_FEATURE_3_COLUMN,
        ]
        + encoded_group_cols,
    )
    return df, df_original
```

#### 2. Descoberta de Estrutura Causal
Utiliza o algoritmo FCI da biblioteca causal-learn para inferir relações causais:

```python
def discover_causal_structure(df: pd.DataFrame, variables: List[str], output_path: str) -> Optional[str]:
    """Discovers the causal structure using FCI and saves the graph."""
    # Implementação do algoritmo FCI para descoberta causal
```

#### 3. Análise SHAP
Quantifica a importância das características usando árvores de decisão:

```python
def calculate_shap_values(df: pd.DataFrame, feature_columns: List[str], target_column: str, output_path: str) -> Optional[str]:
    """Calculates and visualizes SHAP values."""
    # Cálculo e visualização de valores SHAP
```

#### 4. Visualizações Diversas
Implementa múltiplas visualizações para análise dos dados:

```python
def create_visualizations(df: pd.DataFrame, df_original: pd.DataFrame, output_path: str, colors: List[str]) -> None:
    """Creates KDE, violin, and radar plots."""
    create_kde_plot(df, ERRORS_COLUMN, TASK_DIFFICULTY_COLUMN, output_path, colors[:2])
    create_violin_plot(df_original, GROUP_COLUMN, ERRORS_COLUMN, output_path, colors)
    create_radar_plot(df_original, ERRORS_COLUMN, PARTICIPANT_ID_COLUMN, output_path, colors[:3])
```

#### 5. Grafo Bipartido
Cria uma representação de grafo bipartido entre participantes e características:

```python
def visualize_bipartite_graph(df: pd.DataFrame, feature_nodes: Dict[str, List[str]], output_path: str, colors: List[str]) -> None:
    """Visualizes the bipartite graph."""
    # Implementação de grafo bipartido usando networkx
```

#### 6. Agente DDQN
Implementa um agente de aprendizado por reforço:

```python
class DDQNAgent:
    def __init__(self, state_dims: List[int], action_dim: int):
        """Initializes the DDQN agent."""
        # Inicialização do agente DDQN
    
    def act(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Chooses an action using an epsilon-greedy policy."""
        # Seleção de ação baseada em política epsilon-greedy
    
    def learn(self, batch: List[Tuple[np.ndarray, int, float, np.ndarray]], gamma: float = 0.99, learning_rate: float = 0.01) -> None:
        """Updates the Q-network using a batch of experiences."""
        # Atualização da rede Q com experiências
```

#### 7. Análise com LLMs
Utiliza modelos de linguagem para gerar insights sobre os dados:

```python
def generate_insights_report(summary_text: str, causal_info: Optional[str], shap_info: Optional[str], kde_desc: str, violin_desc: str, radar_desc: str, bipartite_desc: str, output_path: str, model_mistral=None, tokenizer_mistral=None, model_grok=None, tokenizer_grok=None, model_grok_enhanced=None, tokenizer_grok_enhanced=None) -> None:
    """Generates an insights report using LLMs (using loaded models)."""
    # Geração de relatório de insights usando modelos LLM
```

#### 8. Fine-tuning de LLM
Implementa ajuste fino de modelos de linguagem usando Unsloth:

```python
model_mistral = FastLanguageModel.get_peft_model(
    model_mistral,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Optimized
    bias="none",  # Optimized
    use_gradient_checkpointing="unsloth",  # Optimized
    random_state=3407,
    use_rslora=False,
)
```

#### 9. Quantização GGUF
Conversão para formato GGUF utilizando llama.cpp:

```python
convert_command = [
    "python3",
    llama_cpp_convert_path,
    hf_model_path,
    "--outfile",
    gguf_output_path,
    "--outtype",
    "q4_0",  # Quantization type
]
```

## Fluxo de Execução

1. Carregamento e validação dos dados
2. Pré-processamento e normalização
3. Descoberta de estrutura causal entre variáveis
4. Cálculo de valores SHAP para importância de características
5. Geração de múltiplas visualizações
6. Criação de grafo bipartido
7. Análise estatística com bootstrap
8. Treinamento do agente DDQN
9. Geração de insights usando LLMs
10. Fine-tuning do modelo Mistral-7B
11. Quantização para formato GGUF

## Saídas

O notebook gera várias saídas, incluindo:
- Gráficos estatísticos (KDE, violin, radar)
- Grafo causal
- Visualização de valores SHAP
- Grafo bipartido
- Relatório de insights baseado em LLM
- Modelo fine-tuned e sua versão quantizada

## Uso

Para executar o notebook, carregue-o em um ambiente Google Colab com acesso a GPU. O notebook está configurado para salvar os resultados em um diretório no Google Drive:

```python
OUTPUT_PATH = "/content/drive/MyDrive/output_errors_mistral_grok/"
```

Certifique-se de modificar este caminho conforme necessário.

## Notas de Implementação

- O código está estruturado com tipagem estática para maior robustez.
- O agente DDQN é implementado de forma simplificada para fins didáticos.
- As funcionalidades de LLM são condicionadas à disponibilidade de GPU.
- É necessário atualizar o caminho para o script de conversão do llama.cpp na seção de quantização GGUF.

## Autor

Hélio Craveiro Pessoa Júnior
