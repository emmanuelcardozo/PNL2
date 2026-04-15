<p  align="left">  
<img  src="https://fi.uba.ar/images/logo-fiuba.png"  alt="FIUBA Logo"  width="320"/>  
</p>

## TinyGPT 

**Curso NLP-II — Facultad de Ingeniería, Universidad de Buenos Aires**  
Autor: Abraham R.

---

## Descripción

TinyGPT es una implementación desde cero de un modelo GPT (*transformer decoder*) de tamaño reducido, diseñado con fines educativos para el posgrado de NLP. El proyecto cubre la arquitectura completa de un GPT moderno, incluyendo una extensión a **Mixture of Experts (MoE)**, equivalente en diseño a modelos como DeepSeek y Mistral.

El modelo se entrena sobre el dataset [Tiny Shakespeare](https://github.com/karpathy/char-rnn) usando un tokenizador a nivel de caracteres.

---

## Arquitectura

```
TinyGPT
├── Token Embedding       (vocab_size → n_embd)
├── Positional Embedding  (block_size → n_embd)
└── N × TransformerBlock
    ├── LayerNorm
    ├── MultiHeadAttention
    │   └── AttentionHead × n_head   (Scaled Dot-Product + KV-cache)
    ├── LayerNorm
    └── FeedForward / MoEFFN
        └── [MoELayer]
            ├── Gate      (Linear → logits por experto)
            ├── Top-k routing
            └── Expert × num_experts  (MLP: Linear → ReLU → Linear → Dropout)
```
---

## Tareas implementadas

### Tarea I — Algoritmos de decodificación (`generateV2`)

Se extiende la función de generación original con tres estrategias aplicables en cascada:

- **Greedy** (`temperature=0`): selecciona el token con mayor probabilidad en cada paso (determinista).
- **Temperature sampling** (`temperature > 0`): escala los logits por `1/T` antes del softmax. Temperatura < 1 = más conservador; > 1 = más creativo.
- **Top-k** (`top_k > 0`): descarta todos los tokens excepto los `k` más probables antes de muestrear.
- **Top-p / nucleus** (`top_p < 1.0`): conserva el menor conjunto de tokens cuya probabilidad acumulada supera `p`, adaptando dinámicamente el tamaño del núcleo.

### Tarea II — Mixture of Experts (MoE)

Se reemplaza la capa FeedForward estándar por una capa **sparse MoE con Top-k routing**:

- **`Expert`**: MLP individual.
- **`Gate`**: red lineal que produce logits de enrutamiento por token.
- **`MoELayer`**: coordina el dispatch de tokens a expertos, normaliza pesos con softmax y acumula salidas ponderadas.
- **`MoEFFN`**: wrapper que implementa la misma interfaz que `FeedForward`, permitiendo intercambiarla sin modificar el resto de la arquitectura.
---

## Estructura del proyecto

```
├── TinyGPT_es.ipynb   # Notebook principal con implementación y experimentos
├── trainer.py         # Módulo de entrenamiento reutilizable
└── README.md
```

---

## Dependencias

```
torch
numpy
tqdm
matplotlib
httpx
```

---

## Referencias

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mistral AI
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy
- [HuggingFace — Text Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [PyTorch AMP Docs](https://docs.pytorch.org/docs/stable/amp.html)
