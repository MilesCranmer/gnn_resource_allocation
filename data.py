# Default units:
#
# Mpc/h, Msun/h, km/s.

# Need to install my astropy branch: https://github.com/MilesCranmer/astropy

import torch
import torch_geometric as tg
import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u
from matplotlib import pyplot as plt
from itertools import product
import torch
from tqdm.auto import tqdm

h = u.littleh
Mpc = u.Mpc
Msun = u.Msun
km = u.km
s = u.s
# -

DATA_DIR = None

if DATA_DIR is None:
    raise ValueError(
        "You need to set the global variable DATA_DIR in data.py to the directory of your halos_*.h5 data."
    )

params = pd.read_csv(
    "latin_hypercube_params.txt", names="Om Ob h ns s8".split(" "), sep=" "
)


def get_raw_data(sim, high_mass_cutoff=3e13):
    global DATA_DIR
    data = pd.read_hdf(DATA_DIR + f"halos_{sim}.h5", "df")
    data.query(f"M14 < {high_mass_cutoff/1e14}", inplace=True)
    return data


def plot_cube(data):
    sample_cube_query = "x < 200 & y < 200 & z < 50"
    sample_cube = data.query(sample_cube_query)
    fig = plt.figure(dpi=150)
    sample_cube.plot(
        "x",
        "y",
        kind="scatter",
        xlabel="x [Mpc/h]",
        ylabel="y [Mpc/h]",
        s=sample_cube.M14 * 100,
        alpha=0.5,
        c="M14",
        cmap="turbo",
        ax=plt.gca(),
        title=f"Halos in ({sample_cube_query})",
    )


def plot_obs_cube(data):
    plt.figure(dpi=150)
    cdata.query("-0.05 < z & z < 0.05").plot(
        "x",
        "y",
        title="Halos within -0.05 < VerticalPositions < 0.05",
        kind="scatter",
        xlabel="x [Mpc/h]",
        ylabel="y [Mpc/h]",
        alpha=0.5,
        s=1,
    )


def get_cosmology_objs(sim):
    cparams = params.iloc[sim]
    actual_h = cparams.h
    cosmo = FlatLambdaCDM(
        H0=100 * cparams.h, Om0=cparams.Om, Tcmb0=2.725, Ob0=cparams.Ob
    )
    h_convert = u.with_H0(
        cosmo.H0
    )  # use with, e.g., (7*Mpc/h).to(Mpc, h_convert) => 10 Mpc
    return cosmo, h_convert


def postprocess(data, cosmo, h_convert, max_redshift=2):
    data = pd.concat(data)
    distances = np.array(data.distance) * Mpc / h
    data["redshift"] = z_at_value(
        cosmo.comoving_distance,
        distances.to(u.Mpc, h_convert),
        zmax=10,
        interpolation="cubic",
        nbins=100000,
    )
    data.query(f"redshift < {max_redshift}", inplace=True)
    return data


def get_data(
    sim,
    starting_point=None,
    subsample=1 / 1000,
    max_redshift=2,
    high_mass_cutoff=3e13,
    direction=None,
):
    data = get_raw_data(sim, high_mass_cutoff=high_mass_cutoff)
    cparams = params.iloc[sim]
    cosmo, h_convert = get_cosmology_objs(sim)

    data = data.sample(frac=1 / 100)
    max_distance = cosmo.comoving_distance(max_redshift).to(Mpc / h, h_convert).value
    cdata = []
    box_width = 1000
    assert np.sqrt(3) * box_width < max_distance
    max_num_copies = int(np.ceil(max_distance / box_width)) + 2
    num_dims = 3
    if starting_point is None:
        starting_point = np.random.rand(3) * box_width
    if direction is None:
        direction = np.random.rand(3)
        direction /= np.sqrt(np.sum(np.square(direction)))

    for dx, dy, dz in product(
        *[list(range(-max_num_copies, max_num_copies + 1))] * num_dims
    ):
        nearest_edge_of_box = (
            np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) - np.sqrt(3)
        ) * box_width
        if nearest_edge_of_box > max_distance:
            continue

        tmp_data = data.sample(frac=subsample * 100)
        tmp_data.x += dx * box_width
        tmp_data.y += dy * box_width
        tmp_data.z += dz * box_width
        displacement = (
            np.array([tmp_data.x, tmp_data.y, tmp_data.z]).T - starting_point[None, :]
        )
        tmp_data["distance"] = np.sqrt(np.sum(displacement ** 2, 1))
        tmp_data.query(f"distance < {max_distance}", inplace=True)
        cdata.append(tmp_data)

    cdata = postprocess(cdata, cosmo, h_convert, max_redshift=max_redshift)
    return cdata, cparams


def get_snr_mapping(snr_map_fname="spec_measurement.npz"):
    snr_mapping = np.load("spec_measurement.npz")
    snr_mapping = pd.DataFrame(
        {
            "zspec": snr_mapping.f.zspec,
            "z": snr_mapping.f.z,
            "time": snr_mapping.f.T,
            "Mspec": snr_mapping.f.Mspec,
            "M": snr_mapping.f.M,
        }
    )
    snr_mapping["zstd"] = snr_mapping.z - snr_mapping.zspec
    snr_mapping["Mstd"] = snr_mapping.M - snr_mapping.Mspec
    return snr_mapping


# Getting dataset of sigma.
#
# Interpolator for predicting sigma from (SNR/z).
#


def assemble_training_data(sims=range(1000), subsample=1e-6):
    all_data = []
    all_cparams = []
    print("Loading")
    for sim in sims:
        data, cparams = get_data(sim, subsample=subsample)

        data["sim"] = sim
        all_data.append(data)
        all_cparams.append(cparams)

    return pd.concat(all_data), np.array(all_cparams)


def get_dataloader(
    sims=range(1000),
    subsample=1e-6,
    batch_size=2,
    cols=["x", "y", "z", "vx", "vy", "vz", "M14", "redshift"],
):
    graphs = []
    for sim in tqdm(sims):
        data, cparams = get_data(sim, subsample=subsample)
        data.x /= 1000
        data.y /= 1000
        data.z /= 1000
        data.vx /= 1000
        data.vy /= 1000
        data.vz /= 1000
        x = data

        graphs.append(
            tg.data.Data(
                x=torch.tensor(np.array(x[cols])).float(),
                y=torch.tensor(cparams).float(),
            )
        )

    graphs = tg.data.DataLoader(graphs, batch_size=batch_size, shuffle=True)
    return graphs
