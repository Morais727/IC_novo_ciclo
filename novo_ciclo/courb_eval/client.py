"""Cliente Flower equivalente ao ClienteFlower de CoUrb-2024/cliente_principal.py.

Mesmo modelo (CNN Keras), otimizador (Adam), 1 época por round e partição de dados
(ManageDatasets, Dirichlet, `dataset_size` amostras por cliente). Diferenças:
  - a escolha de comportamento é dirigida pela config do round enviada pelo servidor
    (`attack`, `malicious`, `pgd_eps`), sem o bug do original
    (`modo=='EMBARALHA' or 'NORMAL' and ...` era sempre verdadeiro);
  - clientes maliciosos em ataques de modelo treinam normalmente: o servidor substitui
    a atualização deles pelo vetor sintetizado (ver strategy.py);
  - dados e modelo ficam em cache por processo (o Flower recria o cliente a cada round).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import flwr as fl
import numpy as np
import tensorflow as tf

from attacks import ATTACK_PARAMS, DATA_ATTACKS, poison_data
from dataset_utils import ManageDatasets

_DATA_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def create_model() -> tf.keras.Model:
    """CNN de CoUrb-2024/cliente_principal.py::cria_modelo (ramo CNN)."""
    m = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def client_data(cid: int, alpha: float, dataset_size: int, seed: int):
    key = (cid, alpha, dataset_size, seed)
    if key not in _DATA_CACHE:
        np.random.seed(seed * 1000 + cid)  # partição reprodutível (o original não fixava semente)
        x, y, _, _ = ManageDatasets(cid, dataset_name="CIFAR10").select_dataset(alpha=alpha, dataset_size=dataset_size)
        _DATA_CACHE[key] = (x.astype(np.float32), y.reshape(-1).astype(np.int64))
    return _DATA_CACHE[key]


def _model() -> tf.keras.Model:
    if "m" not in _MODEL_CACHE:
        _MODEL_CACHE["m"] = create_model()
    return _MODEL_CACHE["m"]


def _pgd_train(model, x, y, w0, radius, seed):
    """Treino Adam com projeção na bola L2 de raio `radius` em torno de w0 após cada passo."""
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    w0 = [tf.constant(w) for w in w0]
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed).batch(32)
    for xb, yb in ds:
        with tf.GradientTape() as tape:
            loss = loss_fn(yb, model(xb, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        diffs = [v - w for v, w in zip(model.trainable_variables, w0)]
        norm = tf.sqrt(tf.add_n([tf.reduce_sum(d ** 2) for d in diffs]))
        if norm > radius:
            for v, w, d in zip(model.trainable_variables, w0, diffs):
                v.assign(w + d * (radius / norm))


class CoUrbClient(fl.client.NumPyClient):
    def __init__(self, cid: int, alpha: float, dataset_size: int, seed: int):
        self.cid = cid
        self.seed = seed
        self.x, self.y = client_data(cid, alpha, dataset_size, seed)

    def get_parameters(self, config):
        return _model().get_weights()

    def fit(self, parameters, config):
        attack = str(config.get("attack", "none"))
        malicious = {int(c) for c in str(config.get("malicious", "")).split(",") if c != ""}
        rnd = int(config.get("server_round", 0))
        is_mal = self.cid in malicious and attack != "none"

        model = _model()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Adam novo
        model.set_weights(parameters)
        tf.random.set_seed(self.seed * 100000 + rnd * 100 + self.cid)

        x, y = self.x, self.y
        if is_mal and attack in DATA_ATTACKS:
            x, y = poison_data(x, y, attack, seed=self.seed + self.cid)

        eps = float(config.get("pgd_eps", 0.0))
        if is_mal and attack == "backdoor_pgd" and eps > 0:
            scale = ATTACK_PARAMS[attack]["scale"]
            _pgd_train(model, x, y, parameters, eps / scale, seed=self.seed + rnd)
            new = [w0 + scale * (w - w0) for w0, w in zip(parameters, model.get_weights())]
        else:
            model.fit(x, y, epochs=1, verbose=0)
            new = model.get_weights()
        return new, len(self.x), {"is_malicious": int(is_mal), "cid": self.cid}
