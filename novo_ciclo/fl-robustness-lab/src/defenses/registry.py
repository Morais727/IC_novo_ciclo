from src.defenses.aggregators import AGGREGATORS
from src.defenses.flame import flame
from src.defenses.layer_guard import layer_guard, layer_guard_sn, layer_guard_client, layer_guard_corr, layer_guard_corr_soft, layer_guard_corr_soft_tm, layer_guard_corr_soft_clip, layer_guard_corr_soft_grad, layer_guard_corr_soft_rand, layerguard, layerguard_cliponly, lg_shadow
from src.defenses.fedsurrogate import fedsurrogate
from src.defenses.repair import fedavg_prune
from src.defenses.dump import fedavg_dump
from src.defenses.trigger_scan import trigger_scan
from src.defenses.foolsgold import foolsgold
from src.defenses.whitening import multi_metric_whitening
from src.defenses.boba import boba
from src.defenses.ragnar import ragnar
from src.defenses.ragnar_original import ragnar_original
from src.defenses.lasa_alignins import lasa, alignins
from src.defenses.rfldefense import rfldefense
from src.defenses.fld_cosdefense import fld, cosdefense

DEFENSE_REGISTRY = {
    **AGGREGATORS,
    "flame": flame,
    "layer_guard": layer_guard,
    "layer_guard_sn": layer_guard_sn,
    "layer_guard_client": layer_guard_client,
    "layer_guard_corr": layer_guard_corr,
    "layer_guard_corr_soft": layer_guard_corr_soft,
    "layer_guard_corr_soft_tm": layer_guard_corr_soft_tm,
    "layer_guard_corr_soft_clip": layer_guard_corr_soft_clip,
    "layer_guard_corr_soft_grad": layer_guard_corr_soft_grad,   # fase 20 (17/11)
    "layer_guard_corr_soft_rand": layer_guard_corr_soft_rand,   # fase 20 (17/11)
    "layerguard": layerguard,
    "layerguard_cliponly": layerguard_cliponly,
    "lg_shadow": lg_shadow,
    "fedsurrogate": fedsurrogate,
    "fedavg_prune": fedavg_prune,
    "fedavg_dump": fedavg_dump,
    "trigger_scan": trigger_scan,
    "foolsgold": foolsgold,
    "multi_metric_whitening": multi_metric_whitening,
    "boba": boba,
    "ragnar": ragnar,
    "ragnar_original": ragnar_original,
    "lasa": lasa,
    "alignins": alignins,
    "rfldefense": rfldefense,
    "fld": fld,
    "cosdefense": cosdefense,
}


def list_defenses():
    return list(DEFENSE_REGISTRY.keys())
