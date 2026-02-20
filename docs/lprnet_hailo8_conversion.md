# Converter LPRNet (`.pth`) para `.hef` (Hailo-8)

Este guia parte do checkpoint já treinado (`Final_LPRNet_model.pth`) e cobre o pipeline recomendado:

1. **PyTorch (`.pth`) -> ONNX (`.onnx`)**
2. **ONNX -> HAR (parse)**
3. **HAR -> Optimized HAR (quantização com calibração)**
4. **Optimized HAR -> HEF (`.hef`)**

> Observação: nomes exatos de comandos podem variar conforme a versão do **Hailo Dataflow Compiler (DFC)**. Se algum comando diferir, consulte `hailo --help` e `hailo <subcommand> --help`.

---

## 1) Pré-requisitos

- Ambiente com PyTorch + ONNX instalados.
- Hailo SDK/DFC instalado (idealmente via ambiente oficial da Hailo).
- Dataset de calibração (imagens representativas do cenário real no Raspberry Pi).

---

## 2) Exportar o modelo para ONNX

Use o script deste repositório:

```bash
python scripts/export_lprnet_to_onnx.py \
  --model-module LPRNet_Pytorch.model.LPRNet \
  --model-class LPRNet \
  --weights LPRNet_Pytorch/weights/Final_LPRNet_model.pth \
  --output build/lprnet.onnx \
  --img-h 24 \
  --img-w 94
```

Se a sua classe/modelo estiver em outro módulo, ajuste `--model-module` e `--model-class`.

Validação rápida:

```bash
python - <<'PY'
import onnx
m = onnx.load('build/lprnet.onnx')
onnx.checker.check_model(m)
print('ONNX OK')
PY
```

---

## 3) ONNX -> HAR (parser)

Exemplo (ajuste para sua versão):

```bash
mkdir -p build/hailo
hailo parser onnx build/lprnet.onnx --hw-arch hailo8 --output-dir build/hailo
```

Resultado esperado: arquivo `.har` (ex.: `build/hailo/lprnet.har`).

---

## 4) Quantização/otimização com calibração

Prepare um diretório com imagens de calibração, por exemplo: `data/calib/`.

Exemplo:

```bash
hailo optimize build/hailo/lprnet.har \
  --calib-set-path data/calib \
  --output-har-path build/hailo/lprnet_optimized.har
```

---

## 5) Compilar para `.hef`

Exemplo:

```bash
hailo compiler build/hailo/lprnet_optimized.har \
  --hw-arch hailo8 \
  --output-hef-path build/hailo/lprnet.hef
```

Arquivo final esperado: `build/hailo/lprnet.hef`.

---

## 6) Dicas importantes para LPR

- Use calibração com placas reais (dia/noite, diferentes ângulos/distâncias).
- Confirme que o pré-processamento no Raspberry Pi é **idêntico** ao do treinamento (resize, normalização, ordem de canais).
- Valide saída numérica do ONNX vs inferência PyTorch antes de compilar para HEF.
- Se houver erros de operadores no parser, exporte com outro `--opset` (11/12/13) e tente novamente.

---

## 7) Troubleshooting rápido

- **"unsupported op" no parser**: reduza/ajuste opset, simplifique o grafo ONNX (`onnxsim`) ou adapte a arquitetura.
- **Perda de acurácia pós-quantização**: aumente qualidade/diversidade da calibração e revise normalização.
- **Mismatch de dimensões**: confirme tamanho de entrada (`24x94` ou o que seu treino usou) em todas as etapas.
