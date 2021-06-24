import torch_geometric as tg
import torch
import sacred
import pickle as pkl
import numpy as np
import matplotlib.patches as patches
from typing import Tuple, Optional, Union
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch_scatter import scatter_mean, scatter_sum
from torch import Tensor
from torch import nn, optim
from pytorch_lightning import seed_everything
from torch_geometric.typing import Adj, Size
from torch_geometric.nn import MetaLayer
from torch.functional import F
from sacred.observers import FileStorageObserver
from sacred import Experiment
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
from importlib import reload
from functools import partial
from accelerate import Accelerator

# Local imports
from model import *
from data import get_data, get_snr_mapping, get_dataloader, get_cosmology_objs, DATA_DIR

# +
ex = Experiment("gnn_allocation")
ex.observers.append(FileStorageObserver("experiments"))

# +
@ex.config
def config():
    high_mass_cutoff = 3e13
    subsample = 1e-3
    max_redshift = 2
    starting_point = None
    version = 2
    clip = 1.0
    lr = 3e-4
    l2reg = 1e-6
    batchsize = 100
    dtau = 0.1
    fixed = True
    shuffle = True

    seed = None
    hid = 32
    nlayers = 1
    nmessages = 3
    total_steps = 30000
    warmup = total_steps / 2

    # Parameters for the model to interpolate (M, time, z) -> (Mstd, zstd)
    snr_map_fname = "spec_measurement.npz"
    snr_hidden = 100
    snr_seed = 0
    snr_clip = 1.0
    snr_lr = 3e-4
    snr_steps = 10000
    snr_batch = 100
    snr_l2reg = 0.0
    snr_verbose = False
    # -

    snr_mapping = get_snr_mapping(snr_map_fname)


# +
# (M, time, z) -> (Mstd, zstd)
# -


@ex.automain
def run(
    _run,
    dtau,
    fixed,
    warmup,
    high_mass_cutoff,
    shuffle,
    subsample,
    max_redshift,
    starting_point,
    clip,
    lr,
    l2reg,
    batchsize,
    snr_map_fname,
    snr_hidden,
    snr_seed,
    snr_clip,
    snr_lr,
    snr_steps,
    snr_batch,
    snr_l2reg,
    snr_verbose,
    snr_mapping,
    hid,
    nlayers,
    nmessages,
    total_steps,
    seed,
):

    y = torch.tensor(np.log10(np.array(np.abs(snr_mapping[["Mstd", "zstd"]])))).float()

    X = torch.tensor(np.array(snr_mapping[["M", "time", "z"]])).float()

    dataset = TensorDataset(X, y)

    train, val = random_split(dataset, [len(X) * 4 // 5, len(X) - len(X) * 4 // 5])

    trainloader = DataLoader(train, num_workers=8, batch_size=64, shuffle=True)
    valloader = DataLoader(val, num_workers=8, batch_size=64)

    snr_model = MLP(3, 2 * 2, nlayers=3, hidden=128)
    opt = optim.Adam(snr_model.parameters(), lr=3e-4)
    num_steps = 200000
    step = 0

    already_trained = True
    try:
        torch.load(open("snr_model.pt", "rb"))
    except FileNotFoundError:
        already_trained = False

    ##### SNR Model:
    if not already_trained:
        while step < num_steps:
            for phase in "train val".split(" "):
                loader = trainloader if phase == "train" else valloader
                losses = []
                if phase == "train":
                    snr_model.train()
                    opt.zero_grad()
                else:
                    snr_model.eval()

                for (x, y) in loader:
                    out = snr_model(x)
                    mu = out[:, 0::2]
                    logvar = out[:, 1::2]
                    loss = ((mu - y) ** 2 / torch.exp(logvar) + logvar) / 2
                    loss = torch.mean(loss)
                    if phase == "train":
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                        #                 sched.step()
                        step += 1
                        if step >= num_steps:
                            break

                    losses.append(loss.item())
                print(
                    phase, np.average(losses), end=(", " if phase == "train" else "\n")
                )
    else:
        snr_model.load_state_dict(torch.load(open("snr_model.pt", "rb")))
    # -

    snr_mapping.time.describe()

    # +
    domain = np.linspace(0, 2, num=100)
    tfilter = 15
    Mfilter = 12
    out = snr_model(torch.tensor([[Mfilter, tfilter, t] for t in domain]).float())
    mu = out[:, 0::2]
    std = torch.exp(out[:, 1::2] / 2)

    snr_mapping["log10(Mstd)"] = np.log10(np.abs(snr_mapping.Mstd))
    # torch.save(snr_model.state_dict(), open('snr_model.pt', 'wb'))

    torch.save(
        snr_model.state_dict(),
        open("snr_model.pt", "wb"),
        _use_new_zipfile_serialization=False,
    )

    plt.figure(dpi=100)

    ax = plt.gca()

    (
        snr_mapping.query(f"abs(M - {Mfilter}) < 0.5")
        .query(f"abs(time - {tfilter}) < 10.0")
        .plot(kind="scatter", x="z", y="log10(Mstd)", s=3, alpha=0.05, ax=ax)
    )

    ax.errorbar(
        domain,
        mu[:, 0].detach().numpy(),
        std[:, 0].detach().numpy(),
        elinewidth=0.3,
        c="k",
    )

    plt.ylim(-5, 0.0)
    # -

    nsims = 2000
    sdata = [
        pkl.load(
            open(
                DATA_DIR + f"graph_obj_{i}.pkl",
                "rb",
            )
        )
        for i in trange(nsims)
    ]

    # +

    for i in range(nsims):
        sdata[i].y = sdata[i].y[None, [0, 4]]  # Om, sigma_8
    # -

    std_params = np.std([sdata[i].y.numpy()[0] for i in range(nsims)], axis=0)
    std_params

    cosmo_objs = [get_cosmology_objs(i) for i in range(nsims)]

    # +

    batch_size = 2
    if fixed:
        dataloader = tg.data.DataLoader(
            sdata[:1800], batch_size=batch_size, shuffle=shuffle
        )
    else:
        dataloader = tg.data.DataLoader(sdata, batch_size=batch_size, shuffle=shuffle)
    # -

    scatter(*dataloader.dataset[0].x[::1000, :2].T)

    HIDDEN = 100

    def random_three_vector():
        """
        Generates a random 3D unit vector (direction) with a uniform spherical distribution
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        :return:
        """
        phi = np.random.uniform(0, np.pi * 2)
        costheta = np.random.uniform(-1, 1)

        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return (x, y, z)

    def angle_to(x1, direction):
        return (
            torch.acos(
                torch.einsum("bx,x->b", x1, direction)
                / torch.sqrt(torch.einsum("bx,bx->b", x1, x1))
            )
            * 180
            / np.pi
        )

    graphgen = partial(tg.nn.knn_graph, k=5)

    field_of_view = 15 / 2  # Actually this is the radius, so divide by 2

    def preprocess_sample(g):
        direction = torch.tensor(random_three_vector()).float().to(acc.device)

        # Define angles relative to these:
        q = torch.tensor(random_three_vector()).float().to(acc.device)
        a1 = torch.cross(direction, q)
        a1 /= torch.norm(a1, p=2)
        a2 = torch.cross(direction, a1)
        xyz = g.x[:, :3]
        d_theta = angle_to(xyz, direction)
        mask = d_theta < field_of_view
        g.x = g.x[mask]
        g.batch = g.batch[mask]
        xyz = g.x[:, :3]
        ra = (angle_to(xyz, a1) - 90) / 180 * np.pi
        dec = (angle_to(xyz, a2) - 90) / 180 * np.pi
        g.x = torch.cat(
            (ra[:, None], dec[:, None], d_theta[mask, None], g.x[:, 3:]),  # (Redundant)
            dim=1,
        ).float()

        return g

    if seed is not None:
        seed_everything(seed)

    gnmodel = GNNAllocation(
        n_in=5,
        n_out=sdata[0].y.shape[1],
        n_v=hid,
        n_e=hid,
        hidden=hid,
        nlayers=nlayers,
        n_messages=nmessages,
    )
    opt = optim.Adam(gnmodel.parameters(), 1e-3)
    sched = optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=total_steps)
    acc = Accelerator()
    clf, opt, load, snrmap = acc.prepare(gnmodel, opt, dataloader, snr_model)

    H = 1000 * 60 * (field_of_view ** 2 / 7.5 ** 2)
    # Most possible: #60 * 100000 * (field_of_view**2/15**2)

    # +
    step = 0
    stepper = iter(trange(total_steps))
    losses = []
    smoothing = 10
    tau = 0.0

    while step < total_steps:
        for g in load:
            g = preprocess_sample(g)

            g.edge_index = graphgen(g.x[:, :2])
            pred, aux = clf(g, snrmap)
            cumulative_time = tg.nn.global_add_pool(aux["time"] - 1, g.batch)

            # Fractional error squared:
            prediction_loss = (
                (pred - g.y) ** 2 / torch.tensor(std_params).to(acc.device)[None] ** 2
            ).mean()
            _run.log_scalar("prediction_loss", prediction_loss)
            time_loss = ((cumulative_time - H) ** 2).mean() / H ** 2
            _run.log_scalar("time_loss", time_loss)
            loss = prediction_loss + time_loss * tau
            if time_loss.item() ** 0.5 > 1e-3 and step > warmup:
                tau += dtau
            _run.log_scalar("tau", tau)

            acc.backward(loss)
            opt.step()
            sched.step()
            step += 1
            if step > total_steps - 1:
                break
            next(stepper)
            opt.zero_grad()
            losses.append([prediction_loss.item(), time_loss.item()])
            if step % smoothing == 0 and step > 0:
                print(np.average(losses[-smoothing:], axis=0))
    # -

    hash_key = str(int(np.random.rand() * 1e8))
    fname = f"gnn_allocator{hash_key}.pt"
    torch.save(
        {
            "state_dict": gnmodel.state_dict(),
            "modelstr": str(gnmodel),
            "params": {"hid": hid, "nlayers": nlayers, "n_messages": nmessages},
        },
        fname,
    )
    _run.add_artifact(fname)

    if fixed:
        testload = acc.prepare(tg.data.DataLoader(sdata[1800:], batch_size=1))
    else:
        testload = acc.prepare(tg.data.DataLoader(sdata, batch_size=1))

    def gen_pretty_figures(g, time=None):

        x = g.x[:, :2].cpu().detach().numpy()
        redshift = g.x[:, -1].cpu().detach().numpy()
        mass = g.x[:, -2].cpu().detach().numpy()

        mask = np.s_[:]  # (redshift - 1.5)**2 < 0.01
        x = x[mask]
        redshift = redshift[mask]
        mass = mass[mask]

        fig = figure(dpi=150, figsize=(8, 8))
        min_alpha = 5
        if time is None:
            alpha = 0.5 * np.ones(mass.shape[0])
        else:
            alpha = (
                0.5
                * (time[:, 0].cpu().detach().numpy()[mask] - 1 + min_alpha)
                / (60.0 + min_alpha)
            )

        c = plt.cm.plasma_r(redshift)
        c[:, -1] = alpha.ravel()
        scatter(
            *x.T, s=mass * 400, c=c, cmap="plasma", linewidths=0.0, vmin=0.0, vmax=2.0
        )

        fig.gca().set_aspect("equal", adjustable="box")
        colorbar().set_label("Redshift")
        w = 0.03
        h = 0.03
        xlow = 0.0
        ylow = -0.05
        rect = patches.Rectangle(
            (xlow, ylow), w, h, linewidth=1, edgecolor="r", facecolor="none"
        )
        xlim(-0.2, 0.2)
        ylim(-0.2, 0.2)
        gca().add_patch(rect)

        fig2 = figure(dpi=150, figsize=(8, 8))

        mask = x[:, 0] > xlow
        mask &= x[:, 0] < xlow + w
        mask &= x[:, 1] > ylow
        mask &= x[:, 1] < ylow + h
        x = x[mask]
        redshift = redshift[mask]
        mass = mass[mask]
        if time is not None:
            print(time.shape, mask.shape)
            alpha = alpha[mask]
        else:
            alpha = 0.5 * np.ones(mass.shape[0])
        c = plt.cm.plasma_r(redshift)
        c[:, -1] = alpha.ravel()

        scatter(
            *x.T,
            s=mass * 100 * (0.6 / w) ** 2,
            c=c,
            cmap="plasma",
            linewidths=0.0,
            vmin=0.0,
            vmax=2.0,
        )
        fig2.gca().set_aspect("equal", adjustable="box")
        colorbar().set_label("Redshift")

        e = graphgen(torch.tensor(x).float())
        plot(x[e, 0], x[e, 1], c="k", alpha=0.1)

        xlim(xlow, xlow + w)
        ylim(ylow, ylow + h)

        return fig, fig2

    j = 0
    for g in testload:
        g = preprocess_sample(g)
        print("True:")
        for i in range(2):
            print("Omega_m sigma_8".split(" ")[i], f"{g.y[0, i].item():.3f}", end=",")
        print()

        print("Predicted:")
        g.edge_index = graphgen(g.x[:, :2])
        pred, aux = clf(g, snrmap)
        for i in range(2):
            print("Omega_m sigma_8".split(" ")[i], f"{pred[0, i].item():.3f}", end=",")
        print()
        print()

        j += 1
        #     break
        if j > 10:
            break
