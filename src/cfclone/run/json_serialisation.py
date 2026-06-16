import json
import networkx as nx
import numpy as np
import pandas as pd


class NumpyPandasEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def serialise_networkx_tree_to_json(nx_tree, json_file):

    tree_data = nx.node_link_data(nx_tree, edges="edges")

    with open(json_file, "w") as f:
        json.dump(tree_data, f, indent=4, cls=NumpyPandasEncoder)


def serialise_dict_to_json(python_dict, json_file):

    with open(json_file, "w") as f:
        json.dump(python_dict, f, indent=4, cls=NumpyPandasEncoder)


def read_json_into_dict(json_file):
    with open(json_file, "r") as f:
        res_dict = json.load(f)
    return res_dict
