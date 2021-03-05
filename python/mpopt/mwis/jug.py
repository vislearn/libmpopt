from mpopt.ct.jug import parse_jug_model
from mpopt.mwis.model import Model


def convert_jug_to_mwis(jug_model):
    models = []
    id_map = {}

    for x in jug_model:
        if x.is_detection():
            if x.timestep == len(models):
                models.append(Model())
            assert x.timestep == len(models) - 1, 'Non-consequetive time step in jug file!'
            v = models[-1].add_node(x.cost)
            assert x.unique_id not in id_map
            id_map[x.unique_id] = (x.timestep, v)
        elif x.is_conf_set():
            tmp = [id_map[unique_id] for unique_id in x.detections]
            t, _ = tmp[0]
            assert all(t_ == t for t_, _ in tmp)
            models[t].add_clique([v for _, v in tmp])
        else:
            assert False, 'Incompatible element type in Jug file for WSP!'

    return models
