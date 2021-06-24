import torch
from torch.functional import F
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer


global_aggregation = scatter_sum
node_aggregation = scatter_sum


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=100, nlayers=2, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.ReLU()]
        for i in range(nlayers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, n_out))
        if layer_norm:
            layers.append(nn.LayerNorm(n_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class EdgeModel(torch.nn.Module):
    def __init__(self, hidden):
        super(EdgeModel, self).__init__()
        self.mlp = MLP(hidden * 4, hidden, layer_norm=True)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        cur_state = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return edge_attr + self.mlp(cur_state)


class NodeModel(torch.nn.Module):
    def __init__(self, hidden):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = MLP(hidden * 2, hidden, layer_norm=True)
        self.node_mlp_2 = MLP(hidden * 3, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = node_aggregation(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return x + self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, hidden):
        super(GlobalModel, self).__init__()
        self.global_mlp = MLP(hidden * 2, hidden, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, global_aggregation(x, batch, dim=0)], dim=1)
        return u + self.global_mlp(out)


class GNN(torch.nn.Module):
    def __init__(self, hidden, n_in=1, n_edge=3, n_out=1, decode_on="node", blocks=5):
        super(self.__class__, self).__init__()
        self.node_enc = MLP(n_in, hidden, layer_norm=True)
        self.edge_enc = MLP(n_edge, hidden, layer_norm=True)
        self.decoder = MLP(hidden, n_out)
        self.ops = nn.ModuleList(
            [
                MetaLayer(EdgeModel(hidden), NodeModel(hidden), GlobalModel(hidden))
                for _ in range(blocks)
            ]
        )
        self.decode_on = decode_on
        self.hidden = hidden

    def forward(self, graph):
        x = self.node_enc(graph.x[:, [3]])  # Only take M14
        pos = graph.x[:, :3]  # Relative position between halos.
        adj = graph.edge_index
        e = self.edge_enc(pos[adj[0]] - pos[adj[1]])

        # Initialize global features as 0:
        u = torch.zeros(
            graph.batch[-1] + 1, self.hidden, device=x.device, dtype=torch.float32
        )
        batch = graph.batch

        for op in self.ops:
            x, e, u = op(x, adj, e, u, batch)

        if self.decode_on == "node":
            out = self.decoder(x)
        elif self.decode_on == "global":
            out = self.decoder(u)

        return out


class GNNAllocation(nn.Module):
    """GNN of the form:
    z_i = f_{in}(x_i)
    For k in Range(n_messages):
        z_i = z_i + g_k(z_i, sum_{j->i} h_k(z_i, z_j))

    y_i = f_{out}(z_i)
    """

    def __init__(
        self,
        n_in,  # e.g., position, Mass
        n_out,  # e.g., Om, s8, etc
        n_v=100,
        n_e=100,
        dim=3,
        hidden=100,
        nlayers=2,
        use_edge_model=False,
        n_messages=5,
        layer_norm=False,
    ):
        super(self.__class__, self).__init__()
        self.allocator = GNN(
            hidden=hidden, n_out=1, decode_on="node", blocks=n_messages
        )
        self.predictor = GNN(
            hidden=hidden, n_out=n_out, decode_on="global", blocks=n_messages
        )

    def forward(self, graph, snr_model):
        orig_graph = graph.clone()

        n = graph.x.shape[0]
        M14 = graph.x[:, [3]].clone()
        true_M = torch.log10(M14 * 1e14)
        true_z = graph.x[:, [4]].clone()
        time1 = torch.ones_like(true_M)
        obs_std1 = snr_model(torch.cat((true_M, time1, true_z), dim=1))
        Mstd1 = torch.exp(np.log(10) * obs_std1[:, [0]])
        zstd1 = torch.exp(np.log(10) * obs_std1[:, [1]])

        graph = orig_graph.clone()
        graph.x[:, [3]] += torch.randn_like(Mstd1) * Mstd1
        graph.x[:, [4]] += torch.randn_like(zstd1) * zstd1

        time2 = (
            time1 + torch.sigmoid(self.allocator(graph) - 3) * 59
        )  # Up to a maximum of 60 minutes per source.

        obs_std2 = snr_model(torch.cat((true_M, time2, true_z), dim=1))
        Mstd2 = torch.exp(obs_std2[:, [0]])
        zstd2 = torch.exp(obs_std2[:, [1]])

        graph = orig_graph
        graph.x = torch.cat(
            (
                graph.x[:, :3],
                graph.x[:, [3]] + torch.randn_like(Mstd2) * Mstd2,
                graph.x[:, [4]] + torch.randn_like(zstd2) * zstd2,
            ),
            dim=1,
        )

        predictions = self.predictor(graph)
        return predictions, {
            "time": time2,
            "Mstd1": Mstd1,
            "zstd1": zstd1,
            "Mstd2": Mstd2,
            "zstd2": zstd2,
        }
