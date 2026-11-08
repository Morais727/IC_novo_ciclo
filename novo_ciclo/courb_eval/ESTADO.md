# courb_eval: estado (pausado em 28/09/2026)

Objetivo: avaliar o RAGNAR no framework do CoUrb-2024 (Flower 1.7 + CNN Keras) contra os 8 ataques
do fl-robustness-lab, comparando o classificador **pré-treinado** do CoUrb
(`CoUrb-2024/MODELOS/{MINMAX,CLASSIFICADOR}_XGB_CNN`) com o **treinado em 5 rounds**.

## Pronto (código escrito, ainda NÃO testado)
- `dataset_utils.py`: cópia do CoUrb (partição Dirichlet), com import opcional de `tensorflow_datasets`
- `attacks.py`: os 8 ataques do laboratório portados para NumPy/Keras
- `client.py`: cliente Flower com a mesma CNN/Adam/1 época do CoUrb, sem o bug `modo=='EMBARALHA' or 'NORMAL'`
- `ragnar_courb.py`: 54 features + MinMax + limiar 0,5 (como `servidor_principal.py`); treino como `XGB_CLASSIFIER.py`
- `strategy.py`: FedAvg + injeção de ataques de modelo + RAGNAR + avaliação centralizada (MA/BA)
- `run_courb.py`: braços fedavg, ragnar_pre, ragnar_5r_nv, ragnar_5r_v (+ clean)

## Parou em: instalar o ambiente
A rede anterior estava lenta demais (~14 s por consulta ao PyPI); o `pip` foi interrompido sem
instalar nada. O `.venv/` existe mas está vazio (só pip). Para retomar:

```bash
cd /home/marcos-morais/Documentos/IC/novo_ciclo/courb_eval
.venv/bin/pip install "flwr==1.7.0" "ray==2.6.3" "pydantic<2" "tensorflow==2.15.0" "keras==2.15.0" \
    "numpy==1.26.4" "xgboost==3.0.5" "scikit-learn==1.7.2" "protobuf==4.25.3" "grpcio==1.60.1" \
    "pandas==2.2.3" "scipy==1.13.1" "matplotlib==3.9.2" 2>&1 | tee install.log
```

## Depois
1. Teste rápido (~2 min): `.venv/bin/python run_courb.py --rounds 3 --attacks lie --arms fedavg ragnar_pre --no-clean --out-dir /tmp/courb_teste`
2. Medir s/round e decidir a escala (40 rounds como no laboratório, ou 20, o padrão do CoUrb).
3. Rodar tudo: `.venv/bin/python run_courb.py` (retomável; pula JSONs já existentes em `results/`).
