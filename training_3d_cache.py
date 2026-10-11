from models.graph_construction import edge_index_to_adj_matrix


class GraphInputCache:
    """Reuse fixed graph inputs across batches, keeping one graph on the device."""

    def __init__(self, device):
        self.device = device
        self._graph = None
        self._inputs = None

    def get(self, graph):
        if graph is not self._graph:
            # Release the previous cache before allocating inputs for another file.
            self._inputs = None
            self._graph = None
            layer_512 = graph['layer_512']
            layer_1024 = graph['layer_1024']
            self._inputs = (
                layer_512.x.to(self.device),
                layer_1024.x.to(self.device),
                edge_index_to_adj_matrix(layer_512).to(self.device),
                edge_index_to_adj_matrix(layer_1024).to(self.device),
                layer_512.y.to(self.device),
                layer_1024.y.to(self.device),
            )
            self._graph = graph
        return self._inputs
