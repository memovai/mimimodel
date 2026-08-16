# mimimodel — un LLM de 45M parámetros con llamada a herramientas, funcionando sin conexión en un ESP32-S3 de 10 $

Un motor de inferencia en C, escrito desde cero en un solo archivo, para
[Needle 2 de Cactus Compute](https://github.com/cactus-compute/needle), ejecutándose por completo
en un microcontrolador ESP32-S3. Sin Linux, sin Python, sin red. Los 13,7 MB de pesos viven en la
flash y **nunca** se cargan en RAM.

[🇺🇸 English](README.md) · [🇯🇵 日本語](README_JA.md) · 🇪🇸 Español · [🇨🇳 中文](README_CN.md)

```
$ turn on pin 5
[{"name":"gpio_on","arguments":{"pin":5}}]
```

| | |
|---|---|
| **Modelo** | Needle 2 — 45M parámetros, CQ de 2 bits, archivo único de 13,7 MB |
| **Hardware** | ESP32-S3, Xtensa LX7 a 240 MHz, 16 MB flash, 8 MB PSRAM (~10 $) |
| **Motor** | un archivo C99, ~2.000 líneas, sin más dependencias que `libm` |
| **Velocidad** | llamada en caliente **29 s** · en frío **241 s** · prefill 1,4 tok/s |
| **Memoria** | 13,7 MB flash (mapeada en memoria) · ~7,7 MB PSRAM · firmware de 256 KB |
| **Precisión** | 15/24 en un benchmark de llamada a herramientas — el motor oficial cerrado saca 14/24 |

> **Honestidad por delante:** esto es unas 5 veces más lento que una API en la nube, no entiende
> chino, y si le saludas te llamará una herramienta igualmente. Véanse las [limitaciones](#limitaciones).
> Lo que ganas es un modelo de lenguaje que funciona con el cable de red desenchufado.

---

## Por qué no bastaba con «compilar el motor para Xtensa»

Cactus publica tanto un [motor de inferencia](https://github.com/cactus-compute/cactus) como el
[modelo Needle](https://github.com/cactus-compute/needle), y su web lista el ESP32-S3 como objetivo
soportado. El plan evidente —compilar el motor de forma cruzada— resultó ser un callejón sin salida,
por dos razones que aparecen al leer el código:

1. **El motor abierto no contiene el núcleo de cómputo de Needle.** Buscar `engram` (un componente
   central de la arquitectura) en todo el repositorio `cactus` no devuelve nada. El motor conoce el
   *nombre* `ModelType::NEEDLE` y sabe formatear sus prompts, pero las capas en sí viven en los
   binarios precompilados por plataforma que se distribuyen en Hugging Face.
2. **Los kernels son solo para ARM.** Los 15 archivos de kernels hacen `#include <arm_neon.h>` y usan
   unos 125 intrínsecos NEON sin ninguna ruta escalar de respaldo. Además, el GCC de Xtensa no
   soporta ni `_Float16` ni `__fp16`, que los kernels emplean unas 770 veces.

Pero *el modelo en sí es completamente abierto*. En `needle/needle/model/` están la arquitectura, el
cuantizador, el bucle de decodificación y —lo decisivo— `export.py`, cuyo docstring es una
especificación completa a nivel de byte del formato de pesos `.cact`. Eso convirtió «escribir un
motor a medida» en el camino correcto, y en uno mucho más pequeño que un runtime de propósito general.

## Cómo funciona

### 1. El formato `.cact`

Una cabecera de 120 bytes lleva toda la geometría de la arquitectura, seguida de los libros de
códigos Lloyd-Max compartidos, un directorio de tensores **sin nombres** (los tensores son
posicionales, en un orden canónico fijo) y luego bloques alineados a 64 bytes. Como la geometría
viaja en la cabecera, un mismo binario carga cualquier configuración de la arquitectura. Analizarlo
son unas 150 líneas de C.

### 2. Cactus Quants, y el truco que mantiene los pesos en la flash

Una matriz `[out, in]` cuantizada con CQ se almacena como índices de 2 bits a un libro de códigos
compartido sobre la esfera unidad, más una norma L2 en fp16 por cada grupo de 128 elementos. La
reconstrucción por grupo es:

```
w_group = (codebook[idx] * norm) @ H        # H = matriz de Walsh–Hadamard normalizada
```

Descuantizar para calcular `w · x` significaría expandir los 13,7 MB enteros en cada token. En su
lugar, el motor aprovecha que `H` es simétrica y ortogonal:

```
(unit · H) · x  ==  unit · (H · x)
```

Así que transformamos la **activación** una sola vez por grupo de 128 con una transformada rápida de
Walsh–Hadamard (O(n log n), 896 sumas), y entonces el producto matriz-vector se reduce a un producto
escalar ponderado por el libro de códigos leído **directamente sobre los bytes empaquetados de 2 bits**.
Los pesos nunca se expanden, nunca se copian y siguen mapeados en memoria desde la flash mediante
`esp_partition_mmap`. Cargar el modelo tarda **48 ms**.

```mermaid
flowchart TB
    subgraph naive ["✗ Ingenuo: descuantizar y luego multiplicar"]
        direction LR
        n1["índices de 2 bits<br/>en la flash"] --> n2["expandir a pesos fp32"]
        n3["normas fp16 por grupo"] --> n2
        n2 --> n4["w · x"]
        n2 -.-> nX["13,7 MB expandidos en cada<br/>token — jamás caben en<br/>512 KB de RAM"]
    end
    subgraph trick ["✓ Identidad de Hadamard: (unit·H)·x ≡ unit·(H·x)"]
        direction LR
        t1["activación x<br/>512 floats"] --> t2["WHT rápida por grupo<br/>de 128 · 896 sumas, una vez"]
        t2 --> t3["producto escalar ponderado<br/>por el libro de códigos"]
        t4["índices de 2 bits — leídos<br/>in situ de la flash mapeada"] --> t3
        t5["normas fp16 por grupo"] --> t3
        t3 --> t6["y = w · x"]
    end
    naive ~~~ trick
```


### 3. Memoria acotada

Needle usa una ventana de atención deslizante de 256 tokens. La caché KV es int8 (el ancho para el
que el modelo fue post-entrenado, según su propia cabecera) y vive en un **búfer circular**
dimensionado a la ventana más un pequeño margen, de modo que la RAM es constante sea cual sea la
longitud del prompt. Una fila solo se sobrescribe `kv_alloc` posiciones más tarde, así que cualquier
`kv_alloc > kv_window` deja intactas todas las filas dentro de la ventana.

```mermaid
flowchart LR
    subgraph FL ["FLASH · 16 MB"]
        F1["firmware<br/>256 KB"]
        F2["partición needle<br/>13,7 MB de pesos"]
    end
    subgraph PS ["PSRAM · 8 MB"]
        P1["búfer circular KV<br/>int8 · 3,3–5,8 MB"]
        P2["estado del modelo<br/>484 KB"]
        P3["caché de pesos<br/>lo que sobre"]
    end
    subgraph SR ["SRAM INTERNA · 512 KB"]
        S1["scratch caliente · 42 KB<br/>x · xh · q/k/v · attn"]
    end
    F2 -- "mmap, lectura in situ<br/>29,9 MB/s" --> S1
    P3 -- "85,5 MB/s" --> S1
    F2 -. "al arrancar: copiar las matrices<br/>más calientes si cabe" .-> P3
    S1 <--> P1
```


### 4. Decodificación restringida por gramática

Un modelo de 45M en carrera libre produce algo *casi* JSON. En vez de eso, el motor guía la
decodificación contra el esquema de herramientas, replicando lo que hace el compilador de gramáticas
del motor cerrado:

- el texto estructural (`[{"name":"`, `","arguments":{`) se **fuerza**, pero enmascarando los logits
  a los tokens que son prefijo de la cadena requerida —nunca insertando IDs de token directamente—,
  de forma que el contexto del modelo se mantiene canónico;
- el **nombre de la herramienta** se elige puntuando la log-probabilidad media completa de cada
  candidato (con teacher-forcing y un rebobinado de contador muy barato), después de que un
  preordenamiento gratuito por el primer token deje solo los 3 mejores candidatos;
- los **argumentos enteros** se enmascaran a dígitos; los **parámetros obligatorios** se fuerzan a aparecer.

```mermaid
stateDiagram-v2
    [*] --> Razonamiento
    Razonamiento --> Razonamiento : tokens de razonamiento libres
    Razonamiento --> Declinado : el modelo emite im_end
    Razonamiento --> Nombre : el modelo emite tool_call
    Nombre --> Args : los 3 mejores candidatos puntuados por log-prob media
    Args --> Entero : parámetro entero — logits enmascarados a dígitos
    Args --> Cadena : parámetro string — libre hasta la comilla de cierre
    Entero --> Bifurcacion
    Cadena --> Bifurcacion
    Bifurcacion --> Args : queda un parámetro obligatorio sin rellenar
    Bifurcacion --> Hecho : todos los obligatorios rellenados
    Hecho --> [*] : JSON siempre válido según el esquema
    Declinado --> [*] : array vacío
```


Por esto el motor supera al oficial en el benchmark pese a usar pesos idénticos: la salida siempre
es válida según el esquema.

### 5. Caché de prefijo KV — la mayor ganancia individual

En un agente de llamada a herramientas, el bloque `<tools>` es idéntico byte a byte en cada llamada
y domina el prompt (288 de 300 tokens). Sus filas KV siguen vivas en el anillo, así que solo hay que
prellenar la consulta. El punto de corte es el marcador `</tools>`: los marcadores son tokens
atómicos, de modo que la tokenización del prefijo es **demostrablemente** un prefijo de la del prompt
completo.

**Llamada en frío 241 s → en caliente 29 s (8,2×).**

```mermaid
flowchart TB
    subgraph C ["Llamada en frío — 241 s"]
        direction LR
        C1["prefill de BOS + bloque de herramientas<br/>288 tokens · 207 s"] --> C2["prefill de la consulta<br/>12 tokens · 9 s"] --> C3["decodificación restringida<br/>~44 pasadas · 25 s"]
    end
    subgraph W ["Llamada en caliente — 29 s · 8,2× más rápida"]
        direction LR
        W1["bloque de herramientas OMITIDO<br/>sus filas KV siguen<br/>vivas en el anillo"] --> W2["prefill de la consulta<br/>12 tokens · 10 s"] --> W3["decodificación restringida<br/>~44 pasadas · 19 s"]
    end
    C -- "instantánea tomada en el marcador &lt;/tools&gt;:<br/>pos · hist_len · epos · anillo engram (40 KB)" --> W
```


## Registro de optimización

Rendimiento bruto del motor, medido sobre hardware real:

| Cambio | Efecto |
|---|---|
| C escalar de partida | prefill 0,64 tok/s, decode 0,59 |
| Reparto en dos núcleos (matvec por filas, atención por cabezas KV) | ~1,8× |
| Omitir la cabeza de logits de 8192×512 durante el prefill | +10% prefill |
| Decodificación de pesos por LUT de byte + kernel de 4 filas (4 filas comparten cada lectura de activación) | +18% |
| Scratch caliente en SRAM interna, no en PSRAM | +5% |
| Caché de pesos en PSRAM (oportunista) | +8% |
| **Total** | **prefill 1,72 tok/s (2,7×), decode 1,38 (2,3×)** |
| Caché de prefijo KV (cuesta ~20% de velocidad bruta por agrandar el anillo) | **8,2× extremo a extremo** |

### Lo que no funcionó

- **SIMD PIE de 128 bits del ESP32-S3.** Implementado en ensamblador (`ee.vmulas.s16.accx`, 8 MAC
  por instrucción) con una ruta de activaciones cuantizadas a int16. Numéricamente correcto
  (error relativo 5,5e-5 en la autoprueba) y **0,32× la velocidad**: 3 veces *más lento*. El cuello
  de botella es desempaquetar los pesos de 2 bits en carriles int16, no las multiplicaciones-acumulaciones,
  y PIE no tiene instrucción de desempaquetado de 2 bits. El código se conserva tras `-DNEEDLE_PIE`,
  desactivado por defecto.
- **Aritmética int16 en el host.** 2,3× más lenta que en float sobre ARM/x86, porque el compilador
  vectoriza automáticamente los bucles en float. SIMD no es una victoria incondicional.
- **Sinkhorn en espacio lineal.** Matemáticamente equivalente a la versión en espacio logarítmico,
  pero sufre underflow y da NaN. Quédate con la logarítmica.

## Limitaciones

- **~29 s por llamada.** Una API en la nube responde lo mismo en 3–8 s. El valor aquí es el
  funcionamiento sin conexión, el coste cero de API y que los datos nunca salgan del dispositivo;
  no la latencia.
- **El chino no funciona.** 0/5 en órdenes de dispositivo en chino. El motor oficial falla igual, así
  que es el límite del modelo, no del port. Al menos su puntuación de confianza cae a 0,02–0,22 en
  estos casos, así que son detectables.
- **No sabe declinar.** Si le pides un chiste, el modelo emite una llamada a herramienta igualmente.
  Cualquier enrutado en producción necesita una puerta de confianza más un prefiltro de texto.
- **Los argumentos booleanos/semánticos no son fiables.** `gpio_write(pin, state)` acierta el `state`
  solo alrededor de la mitad de las veces. Dividirlo en `gpio_on(pin)` / `gpio_off(pin)` —ajustándose
  a lo que el modelo realmente hace bien: elegir nombres y extraer enteros— lleva la precisión en
  escritura de 1/5 a 5/6.
- **La cabeza de confianza aún no está implementada.** Sus pesos están presentes en el archivo
  `.cact`; el motor omite de momento las probe heads.

## Compilar y ejecutar

### En un host (macOS / Linux)

```bash
mkdir -p model && cd model
curl -LO https://huggingface.co/Cactus-Compute/needle2/resolve/main/needle2.cact
cd ..
cc -O3 -o needle needle.c -lm
./needle model/needle2.cact "Set a timer for 10 minutes" \
  '[{"name":"set_timer","description":"Set a countdown timer","parameters":{"type":"object","properties":{"minutes":{"type":"integer","description":"Minutes"}},"required":["minutes"]}}]'
# [{"name":"set_timer","arguments":{"minutes":10}}]
```

Usa `NEEDLE_FREE=1` para decodificación sin restricciones y `NEEDLE_REPEAT=n` para ejercitar la
caché de prefijo.

### En el ESP32-S3

Requiere ESP-IDF v5.5+ y una placa con 16 MB de flash y 8 MB de PSRAM.

```bash
cd needle-esp32s3
idf.py set-target esp32s3
idf.py build
./scripts/flash_weights.sh /dev/ttyUSB0        # 13,7 MB en la partición cruda `needle` en 0x210000
idf.py -p /dev/ttyUSB0 flash monitor
```

Al arrancar ejecuta una llamada a herramienta de demostración y luego entra en un REPL por serie:
escribe una consulta y pulsa Enter.

> ⚠️ Graba la app **antes o junto con** los pesos. Cualquier firmware cuya tabla de particiones
> coloque una región SPIFFS sobre la zona de pesos la formateará automáticamente en el primer
> arranque y corromperá el modelo en silencio (esto nos costó una tarde: la flash devolvía `0xFFFF`,
> que interpretado como coma flotante es `NaN`).

### Archivos

| Ruta | Qué es |
|---|---|
| `needle.c` | el motor: parser, kernels, tokenizador, decodificador restringido, CLI |
| `needle_np.py` | implementación de referencia en numpy, validada contra la decodificación JAX oficial |
| `needle-esp32s3/` | proyecto ESP-IDF (tabla de particiones, grabador de pesos, demo REPL) |
| `bench/` | benchmark de 24 casos de llamada a herramientas, puntuado contra el motor oficial |

### Entorno de desarrollo

El motor en sí no tiene dependencias. La implementación de referencia y el benchmark sí:

```bash
# la referencia en numpy y el contraste con JAX leen el código del paquete original
git clone https://github.com/cactus-compute/needle
pip install numpy sentencepiece jax flax

# el benchmark además usa el motor oficial cerrado como oráculo
pip install cactus-needle
```

`needle_np.py` es una implementación independiente del modelo en numpy. `compare_jax.py` la
compara contra el bucle de decodificación oficial en JAX dentro del árbol `needle/` clonado: esa
es la comprobación que detectó los dos errores descritos en [Corrección](#corrección).

## Benchmark

`bench/run_bench.py` ejecuta 24 casos (herramienta única, extracción de argumentos, desambiguación
entre varias herramientas, chino, órdenes compuestas, charla informal) contra este motor y contra la
biblioteca oficial cerrada usada como oráculo.

```
=== ours 15/24 (62%) | oracle 14/24 (58%) | tool-choice agreement 16/24 (66%)
```

Ambos motores fallan en lo mismo —chino, órdenes compuestas, declinar la charla informal—, que es la
firma esperable de un modelo de 45M compartido y no de una diferencia de implementación.

## Corrección

El motor no se escribió esperando que funcionara. Se construyó como una cadena de equivalencias
verificadas:

1. `needle_np.py` (numpy) se comparó **posición a posición y logit a logit** con el `_forward_cached`
   oficial en JAX del paquete `needle`. Diferencia máxima: 3e-4.
2. `needle.c` se comparó del mismo modo contra `needle_np.py`.
3. En el dispositivo, el firmware autocomprueba el kernel SIMD contra el escalar al arrancar.

```mermaid
flowchart LR
    A["JAX oficial<br/>needle/model/decode.py"] -- "diff posición a posición<br/>y logit a logit · máx 3e-4" --> B["needle_np.py<br/>referencia en numpy"]
    B -- "mismo método de diff" --> C["needle.c<br/>en el host"]
    C -- "autoprueba al arrancar<br/>contra el kernel escalar" --> E["needle.c<br/>en el ESP32-S3"]
```


Dos errores aparecieron solo gracias a esto: los tensores mHC `a_pre`/`a_post`/`a_res` son *escalares*
por capa (el broadcasting de numpy lo ocultaba; en C era una lectura fuera de rango), y los `taps` del
engram son vectores `(4, 512)` por canal, no 4 escalares.

## Créditos

Este proyecto es un *motor*. El modelo, la arquitectura, el esquema de cuantización y el formato son
trabajo de otras personas.

- **[Cactus Compute](https://cactuscompute.com)** — el [modelo Needle 2](https://huggingface.co/Cactus-Compute/needle2)
  (Apache-2.0), el [paquete Python `needle`](https://github.com/cactus-compute/needle) (MIT), cuyos
  `export.py` / `decode.py` / `architecture.py` son la especificación que este motor implementa, y el
  [motor `cactus`](https://github.com/cactus-compute/cactus). Cactus Quants y el formato `.cact` son suyos.
- **Ndubuaku, H., Mosoyan, K., Mroz, J., Cylich, N., Kumar, S., Sandhu, P., Shemet, R. y Lee, J. H.**
  *A Controlled Study of Attention-Only Transformers.* [arXiv:2607.18363](https://arxiv.org/abs/2607.18363)
  — la arquitectura Simple Attention Network sobre la que se construye Needle.
- **[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)** — trabajo previo que demostró un LLM de
  28,9M parámetros en un ESP32-S3 a 9,9 tok/s usando Per-Layer Embeddings. Es lo que hizo que esto
  pareciera merecer el intento.
- **[Andrej Karpathy, llama2.c](https://github.com/karpathy/llama2.c)** — el motor de inferencia en C
  de un solo archivo y sin dependencias al que este se parece.
- **[Espressif](https://github.com/espressif/esp-idf)** — ESP-IDF y
  [esp-dsp](https://github.com/espressif/esp-dsp), cuyo `dspi_dotprod_s16_aes3.S` fue la referencia
  funcional para la sintaxis de las instrucciones vectoriales PIE del ESP32-S3.
- **[SentencePiece](https://github.com/google/sentencepiece)** — el modelo de tokenizador BPE del que
  el blob de tokenizador de `.cact` es un volcado.
- Las transformadas de Walsh–Hadamard y la cuantización de Lloyd-Max son clásicas; la aplicación
  concreta de la identidad de Hadamard para evitar la descuantización es diseño de Cactus, descrito
  en `export.py`.

## Licencia

El código del motor de este repositorio está bajo licencia MIT. Los pesos de Needle 2 son Apache-2.0
y **no** se redistribuyen aquí: descárgalos de Hugging Face. Los repositorios `needle` y `cactus`
conservan sus propias licencias.
